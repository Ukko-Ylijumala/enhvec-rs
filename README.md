# EnhVec

An enhanced Vec implementation for Rust, providing statistical operations, stable hashing, and (somewhat) intelligent sorting management.

## Features

- **Smart Sorting**: Maintains sorting state for optimization and provides methods for sorted access
- **Statistical Operations**: Has statistical methods for numeric vectors
- **Stable Hashing**: Consistent hashing of vectors regardless of element order
- **Enhanced Vector Operations**: Optimized push/insert operations and functional-style iteration helpers
- **Type-Specific Implementations**: Separate implementations for integer and floating-point numbers

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
enhvec = { git = "https://github.com/Ukko-Ylijumala/enhvec-rs" }
```

## Basic Usage

```rust
use enhvec::{EnhVec, Sorting};

// Create from existing data
let mut vec = EnhVec::from_iter(vec![3, 1, 4, 1, 5, 9, 2, 6]);

// Sort and manipulate
vec.push(7);
vec.push_swap_front(8); // O(1) alternative to push_front
vec.sort(Sorting::Ascending);

// Access statistics
println!("Mean: {:?}", vec.average());
println!("Median: {:?}", vec.median());
println!("Standard Deviation: {:?}", vec.stdev());

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
- `from_iter()` - Create from any iterable

### Vector Operations

- `push()`, `push_front()`, `push_swap_front()`, `insert()`, `insert_sorted()`
- `reverse()`, `sort()`
- `for_each()`, `for_each_if()`, `modify_each()`, `modify_each_if()`

### Statistical Operations (Numeric Types)

- Basic: `sum()`, `average()`, `product()`, `range()`
- Distribution: `median()`, `mode()`, `variance()`, `stdev()`, `percentile()`
- Set operations: `distinct()`

### Hashing

- Stable hashing via `xxh3_digest()`

## Type Support

EnhVec supports all standard Rust numeric types through traits:

- `Integer`: For u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
- `Float`: For f32, f64

## License

Copyright (c) 2024-2025 Mikko Tanner. All rights reserved.

License: MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Version History

- 0.3.2: Initial library version
    - Extract EnhVec to a separate library

This library started its life as a component of a larger application, but at some point it made more sense to separate the code into its own little project and here we are.
