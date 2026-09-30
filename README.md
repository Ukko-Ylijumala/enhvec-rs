# EnhVec

An enhanced Vec implementation for Rust, providing statistical operations, stable hashing, and (somewhat) intelligent sorting management.

## Features

- **Smart Sorting**: Tracks the sorting state, so sorted data is not re-sorted, and order-based statistics on sorted data are `O(1)`
- **Statistical Operations**: Has statistical methods for numeric vectors
- **Stable Hashing**: Consistent hashing of vectors regardless of element order
- **Enhanced Vector Operations**: Amortized `O(1)` pushes to the front, sorted inserts, set operations and functional-style iteration helpers
- **Type-Specific Implementations**: Separate implementations for integer and floating-point numbers

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
enhvec = { git = "https://github.com/Ukko-Ylijumala/enhvec-rs" }
```

Without further keys Cargo uses the latest commit of the default branch. A `version` key does
not select a matching release, it only checks the version found there. To pin a release, use
its tag:

```toml
[dependencies]
enhvec = { git = "https://github.com/Ukko-Ylijumala/enhvec-rs", tag = "v0.5.0" }
```

## Basic Usage

```rust
use enhvec::{EnhVec, Sorting};

// Create from existing data
let mut vec = EnhVec::from_iter(vec![3, 1, 4, 1, 5, 9, 2, 6]);
let squares: EnhVec<u64> = (1..=10).map(|x| x * x).collect();

// Sort and manipulate
vec.push(7);
vec.push_front(0); // amortized O(1), keeps the order of the other elements
vec.sort(Sorting::Ascending);

// Access statistics
println!("Mean: {:?}", vec.average());
println!("Median: {:?}", vec.median()); // O(1), as the data is known to be sorted
println!("Standard Deviation: {:?}", vec.stdev());
println!("Product: {:?}", squares.product()); // None on overflow

// Get sorted references without modifying original
let sorted_desc_refs = vec.as_sorted_desc();

// Insert in sorted order
vec.insert_sorted(2);

// Get unique elements
let unique = vec.distinct(Some(Sorting::Ascending));
```

## API Highlights

### Creation Methods

- `new()`, `new_sorted()`, `new_with_capacity()`, `new_from()`
- `from_iter()` / `.collect()` - Create from any iterable, `From<Vec<T>>`

### Vector Operations

- `push()`, `push_front()`, `push_swap_front()`, `insert()`, `insert_sorted()`
- `pop()`, `pop_front()`, `swap_pop_front()`, `remove()`, `truncate()`, `clear()`
- `retain()`, `dedup()`, `drain()`, `split_off()`
- `extend()` (the `Extend` trait, also from `&T` for `Copy` types), `extend_from_slice()`, `extend_sorted()`, `append()`
- `capacity()`, `reserve()`, `reserve_front()`, `shrink_to_fit()`
- `reverse()`, `sort()`, `sort_by()`, `sort_unstable()`, `sort_unstable_by()`, `sort_fp()`, `is_sorted()`, `as_sorted_asc()`, `as_sorted_desc()`
- `get()`, `get_mut()`, indexing, `first()`, `last()`, `to_vec()`, `into_vec()`
- `as_slice()`, `as_mut_slice()`, `Deref`/`DerefMut` to `[T]`, `AsRef<[T]>`/`AsMut<[T]>`, indexing with ranges (`ev[1..3]`), `binary_search()`
- `iter()`, `iter_mut()` (double-ended, e.g. `iter().rev()`), `IntoIterator` for `EnhVec` and its references
- `for_each()`, `for_each_if()`, `modify_each()`, `modify_each_if()`, `count()`, `contains()`

`push_swap_front()` never moves elements to make room at the front: without free space there, the
new element takes the first place and the element there moves to the back. So the order of the
other elements is not preserved. `swap_pop_front()` is the same as `pop_front()`, which is now
`O(1)` in every call.

The elements are always one slice, in order, so all slice methods work on an `EnhVec` (e.g.
`windows()`, `chunks()`, `split_at()`), and `&EnhVec<T>` coerces to `&[T]`. Mutable access
(`as_mut_slice()`, mutable slice methods or indexing) resets the known sort order.
`binary_search()` follows the known order, so it also works on data sorted in descending order.

With a known order (after `sort()` or, for floats, `sort_fp()`, or found by `insert_sorted()`),
`contains()`, `count()` and `binary_search()` are `O(log N)`, and `mode()`, `distinct()` and
`set_relation_sorted()` work on the elements in place, without hashing or sorting: ~600x faster
lookups in 1M elements, and 3-18x for the rest.

The removals keep the order of the other elements, so a known order stays known.
`remove()` moves the shorter part of the other elements, like `VecDeque`. `drain()` moves the
elements out right away, into a Vec of their own, so its iterator does not borrow the `EnhVec`.

`capacity()` counts the free room at both ends. `reserve()` makes room at the back and
`reserve_front()` at the front, so that as many pushes there neither reallocate nor move elements.
`shrink_to_fit()` releases the room at both ends, which is worth it after growing at the front,
where small buffers grow 8x. `extend_from_slice()` copies like `Vec::extend_from_slice()`, where
`extend()` pushes one element at a time (~4x slower for 10k `u64`s).

`sort_unstable()` and `sort_unstable_by()` may reorder equal elements, but are ~1.5x faster and
allocate nothing. For numbers, and other types whose equal elements are indistinguishable, the
result is the same as with `sort()`.

### Set Operations

Values are compared as sets, i.e. order and duplicates are ignored. Only `PartialEq` is
required, so these are `O(N * M)`, but the lookups in an `EnhVec` with a known order are binary
searches: `O(M log N)`.

- `contains_all()`, `contains_any()`, `contains_only()`
- `is_subset()`, `is_superset()`, `is_proper_subset()`, `is_proper_superset()`, `is_proper()`
- `is_equal()`, `is_disjoint()`, `is_partial_overlap()`

Any two non-empty sets are in exactly one of these relations: equal, proper subset, proper
superset, disjoint or partial overlap.

For larger data, compute a `SetRelation` once and check any of the predicates above on it:

- `set_relation()`: `PartialEq` only, `O(N * M)`
- `set_relation_hashed()`: for `Eq + Hash` types, expected `O(N + M)`
- `set_relation_sorted()`: for `Ord` types, `O(N + M)` if both orders are known, else `O(N log N + M log M)`

```rust
let rel = a.set_relation_hashed(&b);
if rel.is_proper_subset() || rel.is_partial_overlap() { /* ... */ }
```

### Statistical Operations (Numeric Types)

- Basic: `sum()`, `average()`, `product()`, `range()`
- Distribution: `median()`, `mode()`, `variance()`, `stdev()`, `percentile()`
- Floating point versions: `median_fp()`, `average_fp()`, `product_fp()`, `variance_fp()`, `stdev_fp()`, `percentile_fp()`
- Unique values: `distinct()`

`variance()` is the population variance (divided by `N`), for integers and floats alike.
`product()` multiplies as `i128` and returns `None` on overflow, and `range()` returns `None` if
the range does not fit in the element type. The floating point versions order NaNs with
`total_cmp()`. `sort_fp()` sorts floats by `total_cmp()` and makes their order known, so that
`median_fp()` and `percentile_fp()` are then `O(1)` (e.g. 3 ms to 40 ns for 1M elements), and
`contains()` and `count()` binary search. Like `==`, those find both -0.0 and +0.0 for either one,
and no NaN. `mode()` breaks ties by first appearance, and `distinct()` keeps the order of first
appearance unless asked to sort, so both are deterministic.

### Hashing

- `Hash`: each element is hashed on its own with xxh3, and the sum and XOR of these digests are
  hashed along with the length. Equal sets of elements hash the same regardless of their order,
  in `O(N)` without sorting or allocating, and the elements need no `Ord`
- Stable hashing via the `Xxh3Hashable` trait: `xxh3()` and `xxh3_digest()`, the same way but
  with each element's `xxh3_digest()`

The element digests do not use the random keys of the hasher given to `Hash`, so whoever controls
the elements can search for colliding sets offline. Keep that in mind before using EnhVecs of
untrusted data as `HashMap` keys.

## Performance

`EnhVec` keeps its elements in order in one buffer, with free space before and after them, so it
grows at both ends. Pushes at both ends are amortized `O(1)` like with `VecDeque`, and pops at
both ends are `O(1)`. When one end runs out of space, the elements are moved back towards the
middle if the buffer is at most half full, or else the buffer grows: 2x at the back like a Vec,
and 8x/4x/2x (tapering with size) at the front, where each growth moves all elements.

Unlike with `VecDeque`, the elements are always one contiguous slice. Indexing costs about the
same as with a Vec, sorting and order statistics work on the slice directly, and `From<Vec<T>>`
and `into_vec()` reuse the buffer (`into_vec()` moves the elements to its start if needed). A
known sort order makes e.g. re-sorting, `median()` and `range()` `O(1)`.

Inserts in the middle shift the shorter side, so like with `VecDeque` at most about half of the
elements. Used as a queue (pushing at one end, popping at the other), the elements drift towards
one end and are moved back now and then, where `VecDeque` wraps around.

`benches/vs_std.rs` compares it to `Vec` and `VecDeque` on common operations. Roughly: pushes at
the back are on par with `Vec`, pushes and pops at both ends are 2-10x faster than with `VecDeque`,
queues 1.1-1.7x faster, and random indexing 1.3-1.6x faster. Indexing and iteration run at `Vec`
speed on data in the cache, but right after building 1M elements at both ends they are slower
(~1.25x and ~2x), as the growing buffer touched more memory than the cache holds. Sorted inserts
are on par with `VecDeque`, and `into_vec()` of data pushed at the front moves it to the start of
the buffer, which is slower than the rearranging of `VecDeque` (~0.7x at 1M elements). Run it with
e.g. `cargo bench --bench vs_std -- pop_front`.

## Type Support

EnhVec supports all standard Rust numeric types through traits:

- `Integer`: For u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
- `Float`: For f32, f64

## License

Copyright (c) 2024-2026 Mikko Tanner. All rights reserved.

License: MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Version History

- 0.6.3: Capacity control, `extend_from_slice()`
    - `capacity()`, `reserve()`, `reserve_front()`, `shrink_to_fit()`, and `extend_from_slice()` as fast as `Vec`'s
- 0.6.2: Vec methods that keep a known order
    - `remove()`, `truncate()`, `clear()`, `retain()`, `dedup()`, `drain()` and `split_off()`
- 0.6.1: Slice API, faster lookups in a known order, `sort_fp()`
    - `Deref`/`DerefMut` to `[T]`, `AsMut<[T]>`, indexing with ranges, `binary_search()` in the known order
    - With a known order: `contains()`, `count()` and the set operations binary search, and `mode()`, `distinct()` and `set_relation_sorted()` need no hashing or sorting
    - `sort_fp()` gives floats a known order: `O(1)` `median_fp()` and `percentile_fp()`, and binary searching `contains()` and `count()`
- 0.6.0: One buffer for all elements, with free space at both ends
    - The elements are always one slice: indexing and iteration run at `Vec` speed, `From<Vec<T>>` and `into_vec()` reuse the buffer
    - Pushes as fast as `Vec::push()`, `O(1)` pops at both ends without moving elements (2-7x faster), faster queues; sorted inserts ~20% slower, as one buffer shifts more elements than two
    - `swap_pop_front()` is the same as `pop_front()`, `push_swap_front()` moves the first element to the back when there is no free space at the front
    - `as_slice()`, `as_mut_slice()` and `AsRef<[T]>`
- 0.5.4: Head in normal order, with free space at both ends
    - Faster sorted inserts, `push_swap_front()` and indexing, `pop()` needs no refill, the head grows 8x/4x/2x as it gets larger
    - `benches/vs_std.rs` keeps freed memory in the process (glibc), so that it measures the containers instead of page faults
- 0.5.3: Faster float sums, sorting and hashing
    - Float sums in `average_fp()`, `variance_fp()` and `variance()` use independent partial sums: ~6x faster on large data
    - `sort_unstable()` and `sort_unstable_by()`, ~1.5x faster than `sort()`, also used by `distinct()`
    - Order-independent hashing from per-element digests instead of a sort: ~25x faster for unsorted data, ~2x for sorted, and `Hash`/`Xxh3Hashable` no longer require `Ord`
    - Breaking: `Hash`/`xxh3` values change, and `custom_xxh3` 0.3.1 or later is required
- 0.5.2: VecDeque-level operations at both ends, Vec/VecDeque comparison bench
    - Amortized `O(1)` `pop_front()` and `pop()` in any mix (two-stack deque), faster indexing and pushes, `benches/vs_std.rs`
    - `insert()`/`insert_sorted()` shift at most about half of the elements (head/main rebalancing)
- 0.5.1: Minor fixes, API gaps and faster set operations
    - `Default` no longer requires `T: Default`, `sort_by()` no longer requires `Ord` (e.g. `sort_by(f64::total_cmp)`)
    - Deterministic `mode()` ties and `distinct()` order, no overflow in `average_fp()`
    - `Extend` (also from `&T`), double-ended, fused and cloneable iterators
    - `SetRelation` via `set_relation()`, `set_relation_hashed()` and `set_relation_sorted()`: all set predicates from one pass, in `O(N + M)` for hashable or sorted data
    - Breaking: `range()` is for `Integer` types and returns `None` if the range does not fit, `Integer`/`Float` have new required methods
- 0.5.0: Correctness and performance pass
    - Fix element order corruption in `compact()`, `insert()`, `reverse()`, `pop_front()` and `insert_sorted()`, and a panic in `insert_sorted()`
    - Fix `sort()`/`extend_sorted()` skipping the sort after mutations, and `iter_mut()` leaving a stale sort state
    - Fix the set operations (`is_subset()` was reversed etc.), add `is_partial_overlap()`, deprecate `is_proper_both()` and `is_proper_either()`
    - Fix overflows in `average()`/`median()`, NaN handling in percentiles and float sorting, make both variances the population variance
    - `PartialEq` compares elements, not the internal layout; add `Eq` and `FromIterator`; add `swap_pop_front()`
    - Amortized `O(1)` `push_front()`, `O(1)`/`O(N)` median, percentile and range, faster `is_sorted()`, `to_vec()`, `into_vec()` and hashing
    - Breaking: `product()` returns `Option<i128>`, `Integer`/`Float` have new required methods, `Hash`/`xxh3` values change (length prefix), `variance_fp()` divides by `N`
- 0.4.0 - 0.4.1: Sorting and extending methods, `IntoIterator` for `EnhVec` and its references
- 0.3.3 - 0.3.9: Internal rework around a separate "head" Vec for front operations, indexing, set operations, `From<Vec<T>>`, `get()`/`get_mut()`, proper iterators, benchmarks
- 0.3.2: Initial library version
    - Extract EnhVec to a separate library

This library started its life as a component of a larger application, but at some point it made more sense to separate the code into its own little project and here we are.
