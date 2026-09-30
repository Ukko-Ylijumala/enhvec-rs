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
- `pop()`, `pop_front()`, `swap_pop_front()`
- `extend()` (the `Extend` trait, also from `&T` for `Copy` types), `extend_sorted()`, `append()`
- `reverse()`, `sort()`, `sort_by()`, `is_sorted()`, `as_sorted_asc()`, `as_sorted_desc()`
- `get()`, `get_mut()`, indexing, `first()`, `last()`, `to_vec()`, `into_vec()`
- `iter()`, `iter_mut()` (double-ended, e.g. `iter().rev()`), `IntoIterator` for `EnhVec` and its references
- `for_each()`, `for_each_if()`, `modify_each()`, `modify_each_if()`, `count()`, `contains()`

`push_swap_front()` and `swap_pop_front()` are `O(1)` in every call (not just amortized), at the
cost of not preserving the order of the other elements.

### Set Operations

Values are compared as sets, i.e. order and duplicates are ignored. Only `PartialEq` is
required, so these are `O(N * M)`.

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
`total_cmp()`. `mode()` breaks ties by first appearance, and `distinct()` keeps the order of first
appearance unless asked to sort, so both are deterministic.

### Hashing

- `Hash`: the elements are hashed in sorted order (plus the length), so equal sets of elements
  hash the same regardless of their order
- Stable hashing via the `Xxh3Hashable` trait: `xxh3()` and `xxh3_digest()`

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

- Unreleased
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
