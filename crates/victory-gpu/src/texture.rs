//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The first of them that is uploaded, and the bytes of title memory
    /// read for it and the smaller ones (title mips).
    first: u32,
    size: usize,
    /// The title's largest mips the host texture leaves out
    /// (`crate::resource::dropped`): host mip `m` is the title's
    /// `m + dropped`, and `first` is never below it.
    dropped: u32,
    /// The bytes of title memory, from the texture's start, each part of
    /// the staging buffer is detiled from (`Layout::subresource_span`),
    /// for a refresh of the pages written alone ([`texture_pages`]).
    /// `None` where parts share memory (3D, thick, multisampled) or for
    /// levels on demand: those are detiled whole.
    spans: Option<Vec<(u64, u64)>>,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list's
    /// prelude last uploaded it, and whether one has uploaded it at all:
    /// `HostList::prelude` records the GPU copy only when either holds,
    /// after `refresh` has already checked title memory for this submission.
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// The memory was found to be view memory, which the guards answer
    /// for: the write watch is not asked about it
    /// (`crate::watch::writes_guarded`).
    guarded: std::sync::atomic::AtomicBool,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// The size of the buffer if it came from the pool of spare staging
    /// buffers and goes back to it ([`staging_buffer`]); 0: a buffer of its
    /// own, released when this is dropped.
    pooled: u64,
    /// Where in `buffer` it begins (the footprints' offsets count it), and
    /// the chunk it is a slot of ([`Chunk`]); `data` maps all of `buffer`.
    base: u64,
    chunk: Option<Arc<Chunk>>,
}

/// A staging buffer nothing holds: a buffer of its own and where it is
/// mapped, or a slot of a chunk and where in it the slot begins.
enum Spare {
    Own(ID3D12Resource, usize),
    Slot(Arc<Chunk>, u64),
}

// SAFETY: a D3D12 buffer and its mapping, touched by nobody while it is
// spare.
unsafe impl Send for Spare {}

/// An upload buffer cut into staging slots of one size, which go back to
/// the spare ones one by one ([`chunk_bytes`]).
struct Chunk {
    buffer: ID3D12Resource,
    data: usize,
    /// Of its `slots`, those spare now (changed under [`SPARE`]'s lock).
    spare: std::sync::atomic::AtomicU32,
    slots: u32,
}

/// The size of the upload buffers staging slots are cut from
/// (`VICTORY_STAGING_CHUNK_MB`; unset or 0 uses individual buffers).
/// The cutscene's longest frames (2 to 4 s) made 1,200 to
/// 4,500 staging buffers each, a `CreateCommittedResource` apiece on the
/// recording thread with the graphics kernel waiting in most of them (some
/// 0.8 ms each); from chunks, one in many slots does.
fn chunk_bytes() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_STAGING_CHUNK_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(0, |n| n.min(256))
            << 20
    })
}

/// Where a staging buffer lies: the buffer, its mapping (of the whole
/// buffer), its pool size (0: not pooled), where in the buffer it begins,
/// and the chunk it is a slot of.
struct Memory {
    buffer: ID3D12Resource,
    data: *mut u8,
    pooled: u64,
    base: u64,
    chunk: Option<Arc<Chunk>>,
}

/// Spare staging buffers by size (a power of two), and their bytes in all.
static SPARE: Mutex<(std::collections::BTreeMap<u64, Vec<Spare>>, u64)> =
    Mutex::new((std::collections::BTreeMap::new(), 0));
/// The most bytes kept spare, and the largest buffer that is pooled.
const SPARE_MOST: (u64, u64) = (256 << 20, 8 << 20);
/// Staging buffers taken from the spare ones, and made, since the start.
pub(crate) static SPARE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A mapped upload buffer of `total` bytes for a texture's staging, all
/// zeros, and its pool size (0: not pooled).
///
/// Every texture the title streams in got a buffer of its own from the
/// host and gave it back when its staging went: about a quarter of a
/// millisecond to make (run 213's profile: 25 ms a frame on the recording
/// threads for some 100 uploads, most of it the graphics kernel waiting)
/// and more to free on the thread that submits. A streaming burst is 200
/// to 500 uploads in a frame (run 222), and those are the frames that
/// dip. Buffers of up to 8 MiB are now kept by size and used again,
/// zeroed as a new one is (levels not there yet must read as zeros).
/// `VICTORY_STAGING_POOL=0`: made and freed each time, as before.
fn staging_buffer(host: &crate::host::Host, total: u64) -> Option<Memory> {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let own = |(buffer, data): (ID3D12Resource, *mut u8), pooled: u64| Memory { buffer, data, pooled, base: 0, chunk: None };
    let size = total.max(64 << 10).next_power_of_two();
    if size > SPARE_MOST.1 || !*ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_POOL").is_none_or(|v| v != "0")) {
        return host.upload_buffer(total).map(|made| own(made, 0));
    }
    let spare = {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = all.0.get_mut(&size).and_then(Vec::pop);
        if let Some(taken) = &taken {
            all.1 -= size;
            if let Spare::Slot(chunk, _) = taken {
                chunk.spare.fetch_sub(1, Relaxed);
            }
        }
        taken
    };
    match spare {
        Some(Spare::Own(buffer, data)) => {
            SPARE_STATS[0].fetch_add(1, Relaxed);
            // SAFETY: the buffer's own mapping, `size` bytes, which nothing
            // else holds.
            unsafe { std::ptr::write_bytes(data as *mut u8, 0, total as usize) };
            return Some(own((buffer, data as *mut u8), size));
        }
        Some(Spare::Slot(chunk, base)) => {
            SPARE_STATS[0].fetch_add(1, Relaxed);
            // SAFETY: the slot's `size` bytes of the chunk's mapping, which
            // nothing else holds.
            unsafe { std::ptr::write_bytes((chunk.data as *mut u8).add(base as usize), 0, total as usize) };
            return Some(Memory {
                buffer: chunk.buffer.clone(),
                data: chunk.data as *mut u8,
                pooled: size,
                base,
                chunk: Some(chunk),
            });
        }
        None => {}
    }
    SPARE_STATS[1].fetch_add(1, Relaxed);
    let bytes = chunk_bytes();
    if bytes < 2 * size {
        return host.upload_buffer(size).map(|made| own(made, size));
    }
    // A new chunk (all zeros, as a new buffer is): its first slot is this
    // one's, the others spare.
    let Some((buffer, data)) = host.upload_buffer(bytes) else {
        return host.upload_buffer(size).map(|made| own(made, size));
    };
    crate::host::name(&buffer, || format!("staging slots of {} KiB", size >> 10));
    let slots = (bytes / size) as u32;
    let chunk = Arc::new(Chunk {
        buffer: buffer.clone(),
        data: data as usize,
        spare: std::sync::atomic::AtomicU32::new(slots - 1),
        slots,
    });
    {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        all.1 += u64::from(slots - 1) * size;
        all.0.entry(size).or_default().extend((1..slots).map(|i| Spare::Slot(chunk.clone(), u64::from(i) * size)));
    }
    Some(Memory { buffer, data, pooled: size, base: 0, chunk: Some(chunk) })
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// The frame its memory was last found changed by the hash alone, with
    /// no write the guard saw (a write through another view of its pages).
    unseen: Option<u64>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Every staging state, grouped by the address where its texture's title
/// memory starts. [`STAGING`] is keyed by resource, which cannot answer
/// "which textures lie in this memory range?"; this index can, so a remap
/// does not have to scan every staged texture.
///
/// Several textures can start at the same address (for example two
/// formats viewing the same memory), hence the `Vec`. Entries are `Weak`
/// so the index never keeps a staging buffer alive on its own; entries
/// whose staging state is gone are removed by [`prune_address_index`].
static STAGED_BY_ADDRESS: Mutex<std::collections::BTreeMap<usize, Vec<Weak<Staging>>>> =
    Mutex::new(std::collections::BTreeMap::new());

/// The size in bytes of the largest texture ever staged. A texture that
/// overlaps some memory range cannot start more than this many bytes
/// before the range, which bounds the search in [`remapped`].
static LARGEST_STAGED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Adds a newly made staging state to [`STAGED_BY_ADDRESS`].
fn add_to_address_index(staging: &Arc<Staging>) {
    let (start, bytes) = staging.memory();
    LARGEST_STAGED_BYTES.fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    let mut index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    let entries = index.entry(start).or_default();
    // A texture staged here before may have been let go since.
    entries.retain(|entry| entry.strong_count() > 0);
    entries.push(Arc::downgrade(staging));
}

/// Removes entries of [`STAGED_BY_ADDRESS`] whose staging state was let go.
fn prune_address_index() {
    let mut index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    index.retain(|_, entries| {
        entries.retain(|entry| entry.strong_count() > 0);
        !entries.is_empty()
    });
}

/// Called when the runtime maps or unmaps title memory `[at, at + len)`.
///
/// A texture in that range may now show completely different bytes, but
/// its staging state still says "uploaded and unchanged". In that state
/// [`upload`] skips the copy and only schedules a check, so the first list
/// that samples the texture after the remap reads the old texels, and the
/// new ones only arrive one list later (#574).
///
/// Marking the texture dirty makes its next [`upload`] record the copy.
/// The check that runs when that list is submitted reads the new memory
/// into the staging buffer before the copy executes, so the very first
/// list after the remap sees the new texels.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let earliest_start = at.saturating_sub(LARGEST_STAGED_BYTES.load(Relaxed));
    let index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    // Textures that start before `end`, and are not so far below `at`
    // that they cannot reach it...
    let candidates = index
        .range(earliest_start..end)
        .flat_map(|(_, entries)| entries.iter().filter_map(Weak::upgrade));
    #[cfg(feature = "diagnostics")]
    let mut examined = 0u64;
    for staging in candidates {
        #[cfg(feature = "diagnostics")]
        {
            examined += 1;
        }
        // ...of which only those whose memory actually reaches past `at`
        // overlap the remapped range.
        let (start, bytes) = staging.memory();
        if start.saturating_add(bytes) > at {
            // The bake probe counts the textures this marks, and those of
            // them that had been found unchanged since their upload.
            #[cfg(feature = "diagnostics")]
            crate::bake_probe::remap_hit(
                start as u64,
                bytes as u64,
                staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed),
            );
            staging.dirty.store(true, Relaxed);
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::bake_probe::remap_call(examined);
}

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    // Host copies with other mips than each other's: not copied.
    if crate::resource::dropped(placed) != crate::resource::dropped(source) {
        return false;
    }
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    let us = started.elapsed().as_micros() as u64;
    REFRESH_WAITS[slot].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    if slot == 2 {
        crate::framelog::add(crate::framelog::T::RefreshStillUs, us);
    }
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
/// Whether a texture the write watch answers for is detiled only where the
/// title wrote since its last refresh: the 64 KiB pages written are read,
/// and the parts of the staging buffer that lie in them detiled from them
/// alone (`VICTORY_TEXTURE_PAGES=1`; off by default: the whole texture is
/// read and detiled at any write, as before).
///
/// After victory-fork's pull request #1130 (by Lkxex), which turns it on
/// by default; it was measured there only in the bytes read and detiled
/// on the CPU, not on a card. Off here until a run shows it: a whole
/// refresh also takes in what changed without a write the watch saw (a
/// remapping), whenever anything in the texture was written; this does
/// not.
pub(crate) fn texture_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_PAGES").is_some_and(|v| v == "1"))
}

/// Since the last frame line: watched refreshes made of the pages written
/// alone ([`texture_pages`]), the bytes they read, and the bytes refreshes
/// of the whole textures would have read.
pub(crate) static PAGE_REFRESHES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // Nothing holds the staging any more (lists keep it until they are
        // reset, after their submission has run): its buffer is spare.
        if self.pooled != 0 {
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(chunk) = &self.chunk {
                all.1 += self.pooled;
                all.0.entry(self.pooled).or_default().push(Spare::Slot(chunk.clone(), self.base));
                // A chunk with every slot spare goes while more than the
                // most kept spare is (released below, outside the lock).
                let spare = chunk.spare.fetch_add(1, Relaxed) + 1;
                if spare == chunk.slots && all.1 > SPARE_MOST.0 {
                    let gone: Vec<Spare> = {
                        let list = all.0.entry(self.pooled).or_default();
                        let (gone, kept) = std::mem::take(list)
                            .into_iter()
                            .partition(|s| matches!(s, Spare::Slot(c, _) if Arc::ptr_eq(c, chunk)));
                        *list = kept;
                        gone
                    };
                    all.1 -= gone.len() as u64 * self.pooled;
                    drop(all);
                    drop(gone);
                }
            } else if all.1 + self.pooled <= SPARE_MOST.0 {
                all.1 += self.pooled;
                all.0.entry(self.pooled).or_default().push(Spare::Own(self.buffer.clone(), self.data as usize));
            }
        }
    }
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let (mut went, mut bytes) = (0u64, 0u64);
    map.retain(|_, s| {
        let keep = Arc::strong_count(s) > 1
            || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
        if !keep {
            LET_GO[0].fetch_add(1, Relaxed);
            LET_GO[1].fetch_add(s.bytes, Relaxed);
            went += 1;
            bytes += s.bytes;
        }
        keep
    });
    if went > 0 {
        prune_address_index();
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        // A layout not worked out takes all memory from its start: no
        // material upload goes over it ([`host_written_overlaps`]).
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(u64::MAX);
        if !at.is_null() && bytes > 0 {
            #[cfg(feature = "diagnostics")]
            crate::bake_probe::note(crate::bake_probe::Writer::HostTexture, at as usize, bytes as usize, 0);
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert((at as u64, resource as usize), (bytes, words[4], words[6], words[7], words[8]));
        }
    }
}

/// The title memory of textures the host copied into, by address and
/// texture: bytes and description words 4, 6, 7 and 8 (width, height,
/// layers and levels, format). Keyed by both, so that a release takes out
/// its own entry without a walk over all of them (after DexNom/VIctory
/// b8ff22e: every release of a resource walked the whole map).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<(u64, usize), (u64, u32, u32, u32, u32)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// Whether a texture the host copied into lies over title memory
/// `[start, end)` (`crate::material_demand`: an upload from title memory
/// there would go over what the card wrote).
pub(crate) fn host_written_overlaps(start: u64, end: u64) -> bool {
    let writes = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
    writes.range(..(end, 0)).rev().any(|(&(base, _), entry)| base.saturating_add(entry.0) > start)
}

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        // The last address at or below, and of its textures one that
        // reaches `address` (one of a layout not worked out says nothing).
        map.range(..=(address, usize::MAX)).next_back().and_then(|(&(at, _), _)| {
            map.range((at, 0)..=(at, usize::MAX)).find_map(|(_, &d)| {
                (d.0 != u64::MAX && address < at.saturating_add(d.0)).then_some((at, d))
            })
        })
    };
    if let Some((at, (bytes, width, height, layers_levels, format))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource whose memory starts at `address`.
pub(crate) fn forget(resource: usize, address: u64) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    drop(map);
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    let wrote = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .is_some_and(|set| set.remove(&resource));
    if wrote {
        HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).remove(&(address, resource));
    }
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Schedules, once per list, the submission-time refresh of `resource`'s
/// staging buffer and (in the list's prelude) its copy into the host
/// texture whenever this is the first upload or `refresh` detiled new
/// contents.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_from(list, resource, texture, 0);
}

/// [`upload`] of the mips from `first` on, the ones a view shows: of a
/// placed texture the title may have no memory for the larger ones, and
/// what lies there is another texture's.
pub(crate) fn upload_from(list: &HostList, resource: This, texture: &ID3D12Resource, first: u32) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // (A host copy without the largest mips has none before its first.)
    let first = first.max(crate::resource::dropped(resource));
    if host_written || !list.first_use(resource as usize) && staged_from(resource) <= first {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture, first) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // Once per submission however many of its lists sample the texture;
    // the list's prelude then records the GPU copy only if `refresh`
    // detiled new contents (or this is the first upload), so unchanged
    // textures skip the copy and changed ones upload on the same submission.
    list.defer_texture(staging.clone());
    // Levels on demand (`crate::tiles`): after the refresh check, in the
    // order of this list's steps, and before the prelude that copies the
    // staging buffer is executed, the levels the staging buffer holds as
    // the title's are given the card's memory if they have none. The
    // prelude copies every level, so each one given here is written whole
    // (the rest are copies into no memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    if let Some(on_demand) = on_demand {
        let staging = staging.clone();
        list.defer_host(move || {
            let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
            staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
            // A texture picked by index has no staging buffer after this
            // upload: its slots are made due again by the texture pass.
            if waiting {
                crate::descriptor::host::levels_waiting(staging.memory as u64);
            }
        });
    }
    list.defer_texture_upload(copy, staging);
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.size)
    }

    /// The host texture's mip and the array slice of the `i`th part of the
    /// staging buffer (the title's mip is `dropped` more).
    fn part(&self, i: usize) -> (u32, u32) {
        let first = self.first - self.dropped;
        let mips = self.mips - first;
        (first + i as u32 % mips, i as u32 / mips)
    }

    /// The texture's subresource the `i`th part of the staging buffer is.
    fn subresource(&self, i: usize) -> u32 {
        let (mip, slice) = self.part(i);
        slice * self.mips + mip
    }

    /// Whether this staging buffer needs to be copied into its host texture
    /// now (first upload or detiled since the last upload), marking it
    /// uploaded. Called from `HostList::prelude` after `refresh` has run.
    pub(crate) fn take_upload(&self) -> bool {
        use std::sync::atomic::Ordering::AcqRel;
        let first = !self.uploaded.swap(true, AcqRel);
        let dirty = self.dirty.swap(false, AcqRel);
        first || dirty
    }

    /// Records the copy of every subresource from the staging buffer into
    /// `texture` (which must be in `D3D12_RESOURCE_STATE_COPY_DEST`).
    pub(crate) fn record_copy(
        &self,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
    ) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: self.subresource(i),
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        let us = whole.elapsed().as_micros() as u64;
        REFRESH_WAITS[3].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        crate::framelog::add(crate::framelog::T::RefreshUs, us);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.size;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        // View memory found so before: asked of the guards alone, below.
        // What it is no view any more is asked as the first time.
        let mut by_guards = None;
        if self.guarded.load(std::sync::atomic::Ordering::Relaxed) {
            by_guards = waited(1, || crate::watch::writes_guarded(start, start + size))
                .map(|(now, written, _)| (now, written));
            if by_guards.is_none() {
                self.guarded.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if by_guards.is_none()
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false))
        {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            let previous = last.asked;
            last.asked = now;
            if previous != 0
                && let Some(read) = self.refresh_pages(&written, previous)
            {
                return read;
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. Writes to the same pages
        // through another view are seen as well (`crate::watch`, links);
        // what the guard still misses is found by the hash, which is taken
        // every eighth frame only, by the texture's address, so the
        // textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // Whether the guard saw no write since the texture was last read.
        let mut blind = false;
        if *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = by_guards.or_else(|| {
                let (now, written, guarded) = waited(1, || crate::watch::writes_how(start, start + size, true))?;
                if guarded && crate::watch::guards_known() {
                    self.guarded.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Some((now, written))
            })
        {
            const HASH_EVERY: u64 = 8;
            // A texture found changed where the guard saw no write is one
            // the title writes through another view of its pages, and it
            // is hashed at every check for as long as it keeps changing: a
            // film plays through a ring of three such textures, one shown
            // per frame, and with each looked at on its own eighth frame
            // the picture went back and forth between moments of the film.
            const UNSEEN_FOR: u64 = 64;
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            let frame = crate::frame::current();
            let turn = frame.wrapping_add((start >> 16) as u64).is_multiple_of(HASH_EVERY);
            let hot = last.unseen.is_some_and(|at| frame < at + UNSEEN_FOR);
            if unwritten && !turn && !hot {
                return 0;
            }
            blind = unwritten;
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_range_now(start, size) else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else {
            waited(2, || crate::watch::title_memory_reading_range(start, size))
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if last.partial.is_some_and(|at| frame < at + 30) {
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = blake3::hash(read);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = blake3::hash(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                last.hash = Some(hash);
                if blind {
                    last.unseen = Some(crate::frame::current());
                }
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        if blind {
            last.unseen = Some(crate::frame::current());
        }
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then nothing is taken and the texture
    /// is looked at again at its next refresh (what was learnt about writes
    /// to it is forgotten, so that it is not passed over as unwritten).
    /// Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = blake3::hash(read);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// A watched refresh of the pages written since the asking `previous`
    /// alone ([`texture_pages`]): each run of written pages is read once
    /// (through the copy, which stands unmapped pages, as a whole refresh
    /// reads), and the parts of the staging buffer that lie in it detiled
    /// from it. Returns the bytes read; `None` where it cannot be done (the
    /// caller refreshes the whole texture).
    fn refresh_pages(&self, written: &[(usize, u64)], previous: u64) -> Option<usize> {
        use std::sync::atomic::Ordering::Relaxed;
        if !texture_pages() {
            return None;
        }
        let spans = self.spans.as_ref()?;
        let start = self.memory as usize;
        let end = start + self.size;
        let page = crate::watch::PAGE;
        // Runs of pages written since, in address order (the journal's).
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for &(p, _) in written.iter().filter(|&&(_, w)| w > previous) {
            let (lo, hi) = ((p * page).max(start), ((p + 1) * page).min(end));
            if lo >= hi {
                continue;
            }
            match runs.last_mut() {
                Some(run) if run.1 == lo => run.1 = hi,
                _ => runs.push((lo, hi)),
            }
        }
        let mut read = 0;
        for &(lo, hi) in &runs {
            with_read(hi - lo, |bytes| {
                crate::host::copy_mapped(bytes.as_mut_ptr(), lo as *const u8, hi - lo, false);
                let (from, to) = ((lo - start) as u64, (hi - start) as u64);
                for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
                    let Some(&(s, e)) = spans.get(i) else {
                        continue;
                    };
                    let (a, b) = (s.max(from), e.min(to));
                    if a >= b {
                        continue;
                    }
                    let (mip, slice) = self.part(i);
                    let f = footprint.Footprint;
                    let pitch = f.RowPitch as usize;
                    let len = pitch * rows as usize * f.Depth as usize;
                    // SAFETY: the subresource's part of the mapped staging
                    // buffer, written only here and by `detile`, under
                    // `last`'s lock.
                    let linear = unsafe { std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len) };
                    if let Err(e) = self.layout.detile_window(
                        &bytes[(a - from) as usize..(b - from) as usize],
                        a,
                        mip + self.dropped,
                        slice,
                        linear,
                        pitch,
                        pitch * rows as usize,
                    ) {
                        tracing::warn!(?e, mip, slice, "texture pages not detiled");
                    }
                }
            });
            read += hi - lo;
        }
        if read != 0 {
            self.dirty.store(true, std::sync::atomic::Ordering::Release);
        }
        PAGE_REFRESHES[0].fetch_add(1, Relaxed);
        PAGE_REFRESHES[1].fetch_add(read as u64, Relaxed);
        PAGE_REFRESHES[2].fetch_add(self.size as u64, Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexWatched, read as u64);
        Some(read)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.dirty.store(true, std::sync::atomic::Ordering::Release);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = self.part(i);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            // (Of a host copy without the title's largest mips, the title's
            // mip is `dropped` more than the host's.)
            if let Err(e) = self.layout.detile(
                tiled,
                mip + self.dropped,
                slice,
                linear,
                pitch,
                pitch * rows as usize,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// Has the next list that uploads a title texture with a staging buffer
/// copy it whole, changed or not since: what the check at a submission
/// finds changed is copied by the next list that samples the texture
/// through a table, and there may be none for one that shaders now pick by
/// index alone.
pub(crate) fn upload_again(resource: This) {
    let staging = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref().and_then(|map| map.get(&(resource as usize)).cloned())
    };
    if let Some(staging) = staging {
        staging.uploaded.store(false, std::sync::atomic::Ordering::Relaxed);
        staging.last.lock().unwrap_or_else(|e| e.into_inner()).partial = None;
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut hasher = blake3::Hasher::new();
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            hasher.update(&read[..bytes]);
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let hash = hasher.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    Some((s.size as u64, committed(s.memory, s.size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are backed by readable memory.
/// The fast path samples console pages. If both ends are absent, walk the
/// virtual-memory regions to distinguish an absent texture from readable
/// mip data in the middle.
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let start = at as usize;
    let end = start.saturating_add(len);
    if start >= end {
        return 0;
    }
    if !by_query() && (committed(at, 1) || committed((end - 1) as *const u8, 1)) {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// The first mip a title texture's staging buffer holds; `u32::MAX` without
/// one.
fn staged_from(resource: This) -> u32 {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .map_or(u32::MAX, |s| s.first)
}

/// The staging state of a title texture from mip `first` on, made on first
/// use (and made anew for larger mips than the one there holds).
fn staging(resource: This, texture: &ID3D12Resource, first: u32) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize))
        && s.first <= first.max(s.dropped)
    {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // A host copy without the title's largest mips begins at the first it
    // has (`d` is the host texture's description, without them).
    let dropped = crate::resource::dropped(resource);
    let wanted = first.max(dropped);
    // Only where the title's layout can end behind the smaller mips.
    let first = if wanted - dropped < u32::from(d.MipLevels) && layout.bytes_from(wanted) < layout.size {
        wanted
    } else {
        dropped
    };
    let size = layout.bytes_from(first) as usize;
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize) - (first - dropped)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first - dropped,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let Memory { buffer, data, pooled, base, chunk } = staging_buffer(host, total)?;
    // A slot of a chunk: its subresources lie past where it begins.
    for footprint in &mut footprints {
        footprint.Offset += base;
    }
    // Where each part lies in title memory (the order of `part`: mips of a
    // slice, slice after slice), for refreshes of written pages alone.
    let spans = (!on_demand && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        .then(|| {
            let mips = u32::from(d.MipLevels) - (first - dropped);
            (0..count)
                .map(|i| layout.subresource_span(first + i % mips, i / mips))
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
    if chunk.is_none() {
        crate::host::name(&buffer, || {
            format!(
                "staging of {resource:?} {}x{}x{} f{}",
                d.Width, d.Height, d.DepthOrArraySize, d.Format.0
            )
        });
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let s = Arc::new(Staging {
        buffer,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        first,
        size,
        dropped,
        spans,
        bytes: total,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        guarded: std::sync::atomic::AtomicBool::new(false),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        pooled,
        base,
        chunk,
    });
    map.insert(resource as usize, s.clone());
    add_to_address_index(&s);
    Some(s)
}

#[cfg(test)]
mod short_tests {
    /// A host copy made short of room leaves out the mips past 256 texels a
    /// side, and its staging buffer begins at the title's first mip it has.
    #[test]
    fn a_texture_made_short_of_room_is_staged_from_the_mips_it_has() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        use windows_sys::Win32::System::Memory::*;
        const BYTES: usize = 2 << 20;
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!base.is_null());
        // A linear 512x512 RGBA8 texture with 4 mips, the title's to fill.
        let description = [3, 0, 0, 0, 512, 0, 512, 0x4_0001, 28, 1, 0, 1, 0, 0];
        let made = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let host_texture = {
                let _short = crate::resource::ShortOfRoom::begin();
                crate::resource::host_resource(resource)?
            };
            Some((resource, host_texture.clone()))
        });
        // `None`: no GPU to make the host texture on.
        if let Some((resource, host_texture)) = made {
            assert_eq!(crate::resource::dropped(resource), 1);
            // SAFETY: a live resource.
            let d = unsafe { host_texture.GetDesc() };
            assert_eq!(d.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE2D);
            assert_eq!((d.Width, d.Height, d.MipLevels), (256, 256, 3));
            let staging = super::staging(resource, &host_texture, 0).expect("staging");
            assert_eq!((staging.first, staging.dropped), (1, 1));
            assert_eq!(staging.footprints.len(), 3);
            assert_eq!(staging.footprints[0].Footprint.Width, 256);
            assert_eq!(staging.part(0), (0, 0));
            assert_eq!(staging.subresource(2), 2);
            assert_eq!(staging.size as u64, staging.layout.bytes_from(1));
            // A view that asks for the title's mip 0 on asks for what it has.
            assert!(std::sync::Arc::ptr_eq(&staging, &super::staging(resource, &host_texture, 0).unwrap()));
            drop(staging);
            super::unstage(resource);
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod remap_tests {
    use std::sync::atomic::Ordering::Relaxed;

    /// After a remap of the memory under a texture that was already
    /// uploaded, the texture must be uploaded again (#574). A remap of
    /// unrelated memory, or of an empty range, must leave it alone.
    #[test]
    fn a_remap_under_a_staged_texture_makes_it_upload_again() {
        use windows_sys::Win32::System::Memory::*;
        const PAGE_RUN: usize = 0x1_0000;
        // Two 64 KiB runs: the texture lives in the first, the second is
        // memory next to it that no texture uses.
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE_RUN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!base.is_null());
        let texture_start = base as usize;
        // Resource description words of a linear 4x4 RGBA8 texture with one
        // mip level, the shape the issue used to reproduce the bug.
        let description = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(texture_start as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::staging(resource, host_texture, 0)
        });
        // `None` means there is no GPU to make the host texture on: nothing
        // to check then.
        if let Some(staging) = staged {
            // The state after a submitted list uploaded the texture.
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);

            super::remapped(texture_start + PAGE_RUN, PAGE_RUN);
            super::remapped(texture_start, 0);
            assert!(!staging.dirty.load(Relaxed), "a remap elsewhere marked the texture");

            super::remapped(texture_start + 0x10, 0x10);
            assert!(staging.dirty.load(Relaxed), "a remap under the texture did not mark it");
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::{mapped_bytes, with_title};

    #[test]
    fn a_streamed_mip_can_be_mapped_between_two_unmapped_ends() {
        use windows_sys::Win32::System::Memory::*;
        const MIP: usize = 64 << 10;
        const SIZE: usize = 4 * MIP;
        // SAFETY: a private reservation; only the committed middle is written.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(mapped_bytes(base, SIZE), 0);
            let middle = base.add(MIP);
            assert_eq!(VirtualAlloc(middle.cast(), MIP, MEM_COMMIT, PAGE_READWRITE), middle.cast());
            let hidden = base.add(2 * MIP);
            assert_eq!(VirtualAlloc(hidden.cast(), MIP, MEM_COMMIT, PAGE_NOACCESS), hidden.cast());
            middle.write_bytes(0xb6, MIP);
            let available = mapped_bytes(base, SIZE);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..MIP].iter().all(|&b| b == 0));
                assert!(bytes[MIP..2 * MIP].iter().all(|&b| b == 0xb6));
                assert!(bytes[2 * MIP..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            assert_eq!(available, MIP as u64);
        }
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }

    /// Creates a small committed 2D `R8G8B8A8_UNORM` linear texture backed by
    /// title memory, for exercising submission-time prelude uploads on the
    /// host D3D12 device.
    fn make_test_texture() -> (
        crate::object::This,
        *mut u8,
        usize,
        &'static windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) {
        let heap_words: [u32; 5] = [1, 0, 0, 1, 1];
        let desc_words: [u32; 14] = [3, 0, 0, 0, 16, 0, 4, 65537, 28, 1, 0, 256, 0, 0];
        // SAFETY: plain structs with exact repr(C) layouts.
        let (heap, desc) = unsafe {
            (
                std::mem::transmute::<[u32; 5], crate::resource::HeapProperties>(heap_words),
                std::mem::transmute::<[u32; 14], crate::resource::ResourceDesc>(desc_words),
            )
        };
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut resource = std::ptr::null();
        let hr = crate::resource::create_committed(
            std::ptr::null(),
            &heap,
            0,
            &desc,
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(hr, crate::object::S_OK);
        let (mem, d) = crate::resource::texture_memory(resource).expect("texture memory");
        let size = crate::resource::texture_layout(&d)
            .expect("console layout")
            .planes[0]
            .layout
            .size as usize;
        let host_tex = crate::resource::host_resource(resource).expect("host texture");
        (resource, mem.cast_mut(), size, host_tex)
    }

    fn read_texel_byte(
        host: &crate::host::Host,
        resource: crate::object::This,
        host_tex: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        extra_list: bool,
    ) -> u8 {
        use windows::Win32::Graphics::Direct3D12::*;
        const COPY_SOURCE: u32 = 0x800;
        // SAFETY: a live host texture.
        let desc = unsafe { host_tex.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: single subresource query into our own stack outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
        }
        let readback = host.read_back_buffer(total).expect("read-back buffer");
        let list = host.new_list().expect("host list");
        super::upload(&list, resource, host_tex);
        crate::list::ensure_state(&list, resource, COPY_SOURCE);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(host_tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: an open list and live resources in COPY_DEST / COPY_SOURCE.
        unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        list.close();
        if extra_list {
            let second = host.new_list().expect("second host list");
            super::upload(&second, resource, host_tex);
            crate::list::ensure_state(&second, resource, COPY_SOURCE);
            second.close();
            host.execute(&[&list, &second]);
        } else {
            host.execute(&[&list]);
        }
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the submission has finished; the read-back buffer is mapped
        // and unmapped within this scope.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).expect("map read-back");
            let byte = *(mapped as *const u8);
            readback.Unmap(0, None);
            byte
        }
    }

    #[test]
    #[ignore = "requires a D3D12 adapter and VICTORY_STAGING_CHUNK_MB=32"]
    fn two_textures_share_one_upload_buffer_without_aliasing() {
        use windows::core::Interface;

        let host = crate::host::host().expect("a host D3D12 device");
        let (first, first_data, first_size, first_texture) = make_test_texture();
        let (second, second_data, second_size, second_texture) = make_test_texture();
        let first_staging = super::staging(first, first_texture, 0).expect("first upload slot");
        let second_staging = super::staging(second, second_texture, 0).expect("second upload slot");

        assert_eq!(first_staging.buffer.as_raw(), second_staging.buffer.as_raw());
        assert_ne!(first_staging.footprints[0].Offset, second_staging.footprints[0].Offset);
        // SAFETY: each pointer maps its own live title texture.
        unsafe {
            first_data.write_bytes(0x31, first_size);
            second_data.write_bytes(0x62, second_size);
        }
        assert_eq!(read_texel_byte(host, first, first_texture, false), 0x31);
        assert_eq!(read_texel_byte(host, second, second_texture, false), 0x62);
        drop((first_staging, second_staging));

        for resource in [first, second] {
            let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
                resource, 0x10,
            );
            assert_eq!(release(resource), 0);
        }
    }

    /// A title-written texture (e.g. a ring-buffered movie frame) that is
    /// updated by the CPU after a clean submission must upload on that same
    /// submission, not a use late.
    #[test]
    fn a_changed_texture_uploads_on_the_same_submission() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (resource, mem, size, host_tex) = make_test_texture();
        // Frame 1: initial contents upload on first submission.
        // SAFETY: `mem` points to `size` bytes of our committed resource.
        unsafe { mem.write_bytes(0x11, size) };
        assert_eq!(read_texel_byte(host, resource, host_tex, false), 0x11);
        // Frame 2: unchanged contents keep the texture as-is (clean state).
        assert_eq!(read_texel_byte(host, resource, host_tex, false), 0x11);
        // Frame 3: CPU writes new contents; the GPU must see them on this
        // same submission rather than sampling the stale previous frame.
        // SAFETY: as above.
        unsafe { mem.write_bytes(0x22, size) };
        assert_eq!(read_texel_byte(host, resource, host_tex, false), 0x22);
        // Release the test resource cleanly.
        let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
            resource, 0x10,
        );
        assert_eq!(release(resource), 0);
    }

    /// When two lists in the same submission sample a title-written texture,
    /// the second list must not clear `dirty` and cause the next submission
    /// after a CPU update to skip its upload.
    #[test]
    fn two_lists_in_one_submission_do_not_drop_the_next_upload() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (resource, mem, size, host_tex) = make_test_texture();
        // SAFETY: `mem` points to `size` bytes of our committed resource.
        unsafe { mem.write_bytes(0x33, size) };
        assert_eq!(read_texel_byte(host, resource, host_tex, true), 0x33);
        // SAFETY: as above.
        unsafe { mem.write_bytes(0x44, size) };
        assert_eq!(read_texel_byte(host, resource, host_tex, false), 0x44);
        let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
            resource, 0x10,
        );
        assert_eq!(release(resource), 0);
    }
}
