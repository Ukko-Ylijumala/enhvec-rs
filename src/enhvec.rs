// Copyright (c) 2024-2025 Mikko Tanner. All rights reserved.

use crate::hashing::{CustomXxh3Hasher, Xxh3Hashable};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    iter::Sum,
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
};

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
/// The expected sorting state of an [EnhVec].
pub enum Sorting {
    #[default]
    None,
    Ascending,
    Descending,
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
/// The current sorting state of an [EnhVec].
enum SortState {
    #[default]
    Unsorted,
    Asc,
    Desc,
    Changed, // changed - could be sorted or not, depending on what happened
    SwappedToFront(usize, Box<SortState>), // push_swap_front() was used N times
}

/**
A wrapper around a Vec of elements (objects/items).

This struct provides additional methods for handling elements:
- sorting the elements ascending or descending
- returning references to the elements, also sorted
- pushing elements to the front of the vector
- hashing the elements in a stable, repeatable way
*/
#[derive(Debug, Default, Clone)]
pub struct EnhVec<T> {
    v: Vec<T>,
    sort: Sorting,
    state: SortState,
}

// Technically PartialEq and PartialOrd bounds are not needed for the
// methods in this block, but we want to restrict the types allowed
// in EnhVec to those that can be compared and sorted.
impl<T: PartialEq + PartialOrd> EnhVec<T> {
    fn default() -> Self {
        Self {
            v: Vec::<T>::new(),
            sort: Sorting::None,
            state: SortState::Unsorted,
        }
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn new_sorted(sorting: Sorting) -> Self {
        Self {
            sort: sorting,
            ..Self::default()
        }
    }
    pub fn new_with_capacity(capacity: usize) -> Self {
        Self {
            v: Vec::<T>::with_capacity(capacity),
            ..Self::default()
        }
    }
    pub fn new_from(elements: Vec<T>) -> Self {
        Self {
            v: elements,
            ..Self::default()
        }
    }
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            v: iter.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Insert an element at index. Possibly slow, as it may shift other elements.
    pub fn insert(&mut self, idx: usize, element: T) {
        self.v.insert(idx, element);
        self.state = SortState::Changed;
    }

    /// Push an element to the end of the [EnhVec].
    pub fn push(&mut self, element: T) {
        self.v.push(element);
        self.state = SortState::Changed;
    }

    /**
    Insert an element at the start of the [EnhVec].

    NOTE: potentially slow, as it must shift all other elements to make
    room for the new first one. Time complexity: `O(N)`. Prefer `push()`
    and finally `sort()` if you need to maintain a certain order, or
    `push_swap_front()` if you just need the new element to be the
    first one and don't particularly care about the rest.
    */
    pub fn push_front(&mut self, element: T) {
        self.insert(0, element);
    }

    /**
    Insert an element at the end of the [EnhVec], then swap it with the
    first element. This is a much faster way to push an element to the
    front of the Vec than `push_front()`, as it doesn't require shifting
    all other elements. Time complexity: `O(1)`.
    */
    pub fn push_swap_front(&mut self, element: T) {
        let last: usize = self.len(); // len() - 1 after push()
        self.v.push(element);
        if last > 0 {
            self.swap(0, last);
        }

        match &self.state {
            SortState::Asc | SortState::Desc => {
                self.state = SortState::SwappedToFront(1, Box::new(self.state.clone()));
            }
            SortState::SwappedToFront(num, prevstate) => {
                self.state = SortState::SwappedToFront(num + 1, prevstate.clone());
            }
            _ => {
                self.state = SortState::Changed;
            }
        }
    }

    /// Reverse the order of the elements in place. [SortState] and [Sorting] are updated.
    pub fn reverse(&mut self) {
        self.v.reverse();
        match self.state {
            SortState::Changed | SortState::Unsorted | SortState::SwappedToFront(_, _) => {
                self.state = SortState::Changed;
                self.sort = Sorting::None;
            }
            SortState::Asc => {
                self.state = SortState::Desc;
                self.sort = Sorting::Descending;
            }
            SortState::Desc => {
                self.state = SortState::Asc;
                self.sort = Sorting::Ascending;
            }
        }
    }

    /// Return a [Vec] of references to entries.
    pub fn as_ref_vec(&self) -> Vec<&T> {
        self.v.iter().collect()
    }
    /// Return a [Vec] of mutable references to entries.
    pub fn as_mut_ref_vec(&mut self) -> Vec<&mut T> {
        self.state = SortState::Changed; // sort state could change
        self.v.iter_mut().collect()
    }

    /// Run a closure on each element.
    pub fn for_each(&self, f: impl FnMut(&T)) {
        self.v.iter().for_each(f)
    }
    /// Run a closure on each element if the predicate is true.
    pub fn for_each_if(&self, mut f: impl FnMut(&T), predicate: impl Fn(&T) -> bool) {
        self.v.iter().for_each(|elem: &T| {
            if predicate(elem) {
                f(elem)
            }
        })
    }

    /// Run a mutating closure for each element.
    pub fn modify_each(&mut self, f: impl FnMut(&mut T)) {
        self.state = SortState::Changed;
        self.v.iter_mut().for_each(f)
    }
    /// Run a mutating closure for each element if the predicate is true.
    pub fn modify_each_if(&mut self, mut f: impl FnMut(&mut T), predicate: impl Fn(&T) -> bool) {
        self.state = SortState::Changed;
        self.v.iter_mut().for_each(|elem: &mut T| {
            if predicate(elem) {
                f(elem)
            }
        })
    }

    /// Clone the elements into a new regular [Vec<T>].
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.v.clone()
    }
    /// Consume the [EnhVec] and return the inner [Vec<T>].
    pub fn into_vec(self) -> Vec<T> {
        self.v
    }
}

impl<T: Ord> EnhVec<T> {
    /**
    Whether the [EnhVec] is sorted in ascending order. Worst case time
    complexity: `O(N)`, as it may have to compare each element with the next.

    NOTE: an empty or 1-element EnhVec is considered sorted.
    */
    pub fn is_sorted(&self) -> bool {
        match self.len() {
            // must check for empty first to avoid panic with `windows()` method
            0 | 1 => return true,
            2 => return self[0] <= self[1],
            _ => {}
        }
        if self.state == SortState::Asc {
            // short circuit if already sorted
            return true;
        }
        if (self.first().unwrap()).gt(&self.last().unwrap()) {
            // short circuit if first > last
            return false;
        }
        self.v
            .iter()
            .zip(self.v.iter().skip(1))
            .all(|(a, b)| a <= b)

        // TODO: check if this is faster than the iter().skip(1)
        // above for large Vecs and optimize accordingly
        //     self.v.windows(2).all(|w| w[0] <= w[1])
    }

    /**
    Insert an element into the [EnhVec] in sorted order.

    NOTE: the Vec must be sorted ASC for this insert to make much sense.
    If the Vec is not sorted, the insertion point would be more or less
    random, hence in this case we just `push()` the element to the end.

    NOTE: this method is potentially slow, as it might traverse the Vec 2 times:
    once to check if it is sorted (worst case: `O(N)`), and once to find the
    insertion point (`O(log n)`). This may be be optimized in the future.

    NOTE: if you need to insert many elements, it will likely be faster to sort
    the Vec after all the insertions are done, as sorting is approx. `O(N log N)`.
    */
    pub fn insert_sorted(&mut self, element: T) {
        if self.is_empty() || element >= *self.last().unwrap() || !self.is_sorted() {
            // short circuit some common cases
            self.v.push(element);
            return;
        }

        let cutoff: usize = 20;
        let idx: usize = match self.len() < cutoff {
            // linear search for "small" vectors
            true => self.iter().position(|x: &T| element < *x).unwrap_or(self.len()),
            // binary search for larger vectors
            false => match self.binary_search(&element) {
                Ok(index) | Err(index) => index,
            },
        };
        self.v.insert(idx, element);
    }

    /// Set the default sorting state of the [EnhVec] and sort the data.
    pub fn sort(&mut self, sorting: Sorting) {
        if sorting == self.sort {
            return;
        }
        sort_vec(&mut self.v, &self.state, sorting);
        self.sort = sorting;
        match sorting {
            Sorting::Ascending => self.state = SortState::Asc,
            Sorting::Descending => self.state = SortState::Desc,
            _ => self.state = SortState::Unsorted,
        }
    }

    /// Return references to entries in ASCending order.
    pub fn as_sorted_asc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        sort_vec(&mut vec, &self.state, Sorting::Ascending);
        vec
    }
    /// Return references to entries in DESCending order.
    pub fn as_sorted_desc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        sort_vec(&mut vec, &self.state, Sorting::Descending);
        vec
    }
}

/* --------------------------------- */

impl<T> Deref for EnhVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> DerefMut for EnhVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl<T: PartialEq> EnhVec<T> {
    /// Count the occurrences of a value.
    pub fn count(&self, value: &T) -> usize {
        self.iter().filter(|&x| x == value).count()
    }
}

impl<T: PartialEq> PartialEq for EnhVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

/* ################## Hashing and custom hashing behaviour ################# */

impl<T: Ord + Hash> Hash for EnhVec<T> {
    /**
    The hash of a Vec is **not** the same as iterating over the elements
    and accumulating the state from each one individually. Per the docs:

    "The hash of a vector is the same as that of the corresponding slice"

    Hence we must implement our own hashing method since we want to be able
    to repeatably produce the same hash from the same set of elements,
    regardless of any other factors. This also means that we must always
    hash the elements in the same (sorted) order, AND that the hash algo
    must be stable (i.e. always produce the same hash for the same input).

    Since the standard hasher is not stable, the output of this method
    will not be the same across different runs of the program, and will
    change each time the standard hasher's [std::hash::RandomState] changes.

    To produce truly repeatable hashes, it is recommended to use the `xxh3()`
    or `xxh3_digest()` methods instead, which use a stable hasher.
    */
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_sorted_asc()
            .iter()
            .for_each(|elem| elem.hash(state));
    }
}

impl<T: Ord + Xxh3Hashable> Xxh3Hashable for EnhVec<T> {
    /**
    This method is used to hash the elements in a stable, repeatable way.
    Internally it works just like the standard `hash()` method, ie. it
    updates the state of the given hasher with each element in turn.

    The element in question must implement the [Xxh3Hashable] trait and
    actually hash itself using the `xxh3()` method of course.
    */
    #[inline]
    fn xxh3<H: Hasher>(&self, state: &mut H) {
        self.as_sorted_asc()
            .iter()
            .for_each(|elem| elem.xxh3(state));
    }

    /**
    This method is used to hash the elements in a stable, repeatable way.
    In contrast to `xxh3()`, this method returns the final u64 hash value.
    */
    #[inline]
    fn xxh3_digest(&self) -> u64 {
        let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
        self.as_sorted_asc()
            .iter()
            .for_each(|elem| elem.xxh3(&mut hasher));
        hasher.finish()
    }
}

/* ###################### EnhVec for numeric elements ###################### */

impl<T> EnhVec<T>
where
    T: Copy + Sum,
{
    /// Return the sum of all elements.
    pub fn sum(&self) -> T {
        self.iter().copied().sum()
    }
}

impl<T> EnhVec<T>
where
    T: Copy + Ord + Sub<Output = T>,
{
    /// Return the range (max - min) of the elements.
    pub fn range(&self) -> Option<T> {
        if let (Some(min), Some(max)) = (self.iter().min(), self.iter().max()) {
            Some(*max - *min)
        } else {
            None
        }
    }
}

/* --------------------------------- */

impl<T> EnhVec<T>
where
    T: Copy + Eq + Hash,
{
    /// Return the mode (most common) value of the elements.
    pub fn mode(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut counts: HashMap<T, u32> = HashMap::new();
        for &item in self.iter() {
            *counts.entry(item).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(item, _)| item)
    }

    /// Return the distinct (unique) elements, optionally sorted.
    pub fn distinct(&self, sorted: Option<Sorting>) -> EnhVec<T>
    where
        T: Copy + Eq + Hash + Ord,
    {
        let set: HashSet<T> = self.iter().copied().collect();
        let mut result: EnhVec<T> = EnhVec::from_iter(set.into_iter());
        if sorted.is_some() {
            result.sort(sorted.unwrap());
        }
        result
    }
}

/* --------------------------------- */

impl<T: Integer> EnhVec<T> {
    /// Return the median (aka. the middle) value of the elements.
    pub fn median(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let sorted: Vec<&T> = self.as_sorted_asc();
        let mid: usize = sorted.len() / 2;

        if sorted.len() % 2 == 0 {
            Some((*sorted[mid - 1] + *sorted[mid]) / T::from_usize(2).unwrap())
        } else {
            Some(*sorted[mid])
        }
    }

    /// Return the average (mean) value of the elements.
    pub fn average(&self) -> Option<f64>
    where
        T: Into<i128>,
    {
        if self.is_empty() {
            return None;
        }

        let sum: i128 = self.iter().copied().sum::<T>().into();
        Some(sum as f64 / self.len() as f64)
    }

    /// Return the product of all elements. For empty EnhVec, `product == 1`.
    /// To avoid overflow, multiplications are performed as `i128`.
    pub fn product(&self) -> i128
    where
        T: Into<i128>,
    {
        self.iter().fold(1, |acc: i128, &x| acc * x.into())
    }

    /// Return the variance of the elements.
    pub fn variance(&self) -> Option<f64>
    where
        T: Into<i128>,
    {
        if self.len() < 2 {
            return None;
        }

        let mean: f64 = self.average()?;
        let variance: f64 = self
            .iter()
            .map(|&x| (x.into() as f64 - mean).powi(2))
            .sum::<f64>()
            / self.len() as f64;
        Some(variance)
    }

    /// Return the standard deviation of the elements.
    pub fn stdev(&self) -> Option<f64>
    where
        T: Into<i128>,
    {
        self.variance().map(|v: f64| v.sqrt())
    }

    /// Return the percentile value of the elements. NOTE: `0.0 <=` [p] `<= 1.0`
    pub fn percentile(&self, p: f64) -> Option<T> {
        if self.is_empty() || p < 0.0 || p > 1.0 {
            return None;
        }

        let sorted: Vec<&T> = self.as_sorted_asc();
        let index: usize = (p * (self.len() - 1) as f64).round() as usize;
        Some(*sorted[index])
    }
}

/* --------------------------------- */

impl<T: Float> EnhVec<T> {
    /// Return the median (aka. the middle) value of the elements.
    /// Floating point compatible version.
    pub fn median_fp(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut sorted: Vec<T> = self.to_vec();
        sorted.sort_by(|a: &T, b: &T| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mid: usize = sorted.len() / 2;

        if sorted.len() % 2 == 0 {
            T::from_usize(2).and_then(|two: T| Some((sorted[mid - 1] + sorted[mid]) / two))
        } else {
            Some(sorted[mid])
        }
    }

    /// Return the average (mean) value of the elements.
    /// Floating point version.
    pub fn average_fp(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let sum: T = self.iter().copied().sum();
        Some(sum / T::from_usize(self.len()).unwrap())
    }

    /// Return the product of all elements. Floating point version.
    pub fn product_fp(&self) -> T {
        self.iter().fold(T::one(), |acc, &x| acc * x)
    }

    /// Return the variance of the elements. Floating point version.
    pub fn variance_fp(&self) -> Option<T> {
        if self.len() < 2 {
            return None;
        }

        let mean: T = self.average_fp()?;
        let variance: T = self.iter().map(|&x| (x - mean).powi(2)).sum::<T>()
            / T::from_usize(self.len() - 1).unwrap();
        Some(variance)
    }

    /// Return the standard deviation of the elements. Floating point version.
    pub fn stdev_fp(&self) -> Option<T> {
        self.variance_fp().map(|v: T| v.sqrt())
    }

    /// Return the percentile value of the elements. Floating point version.
    /// NOTE: `0.0 <=` [p] `<= 1.0`
    pub fn percentile_fp(&self, p: f64) -> Option<T> {
        if self.is_empty() || p < 0.0 || p > 1.0 {
            return None;
        }

        let mut sorted: Vec<T> = self.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let index: usize = (p * (self.len() - 1) as f64).round() as usize;
        Some(sorted[index])
    }
}

/* ################# Traits and impls for numeric elements ################# */

/// [EnhVec] type trait for all integers.
pub trait Integer:
    Copy
    + Eq
    + Ord
    + PartialEq
    + PartialOrd
    + Hash
    + Sum
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(n: usize) -> Option<Self>;
}

/// Common code for "small" integer types.
macro_rules! impl_integer {
    ($($t:ty),*) => {
        $(
            impl Integer for $t {
                #[inline]
                fn zero() -> Self { 0 }
                #[inline]
                fn one() -> Self { 1 }
                #[inline]
                fn from_usize(n: usize) -> Option<Self> { n.try_into().ok() }
            }
        )*
    }
}

/// Common code for "big" integer types.
macro_rules! impl_big_integer {
    ($($t:ty),*) => {
        $(
            impl Integer for $t {
                #[inline]
                fn zero() -> Self { 0 }
                #[inline]
                fn one() -> Self { 1 }
                #[inline]
                fn from_usize(n: usize) -> Option<Self> { Some(n as Self) }
            }
        )*
    }
}

impl_integer!(u8, u16, u32, i8, i16, i32, isize);
impl_big_integer!(u64, u128, usize, i64, i128);

/* --------------------------------- */

/**
In contrast to [Integer], we must remove [Eq] and [Ord] constraints, as
they are not defined for floating point numbers due to `NaN`. Also [Hash]
is not implemented for f32/f64, so we must remove that constraint as well.

The `f32` and `f64` types are IEEE 754 floating point numbers, which are not
exact representations of real numbers. Hence floating point arithmetic is not
exact and can (will) lead to rounding errors. For example, `0.1 + 0.2` is very
close to `0.3`, but not exactly equal to it. This is due to the fact that
floating point numbers cannot be exactly represented in binary and the result
is a number that is extremely close to `0.3`, but not quite.

This difference is usually denoted as a (very small) number called the "machine
epsilon" (`ε`), which is the smallest number that can be added to `1.0` to get a
result different from `1.0`.

## `f32`
- range: ±3.40282 × 10^38
- smallest normal: 1.17549 x 10^-38
- smallest subnormal: 1.4 × 10^−45
- precision: ~7 decimal digits (≈1.19 × 10^−7)
## `f64`
- range: ±1.79769 × 10^308
- smallest normal: 2.22507 x 10^-308
- smallest subnormal: 4.94 × 10^−324
- precision: ~15-17 decimal digits (≈2.22 × 10^−16)
### Both types can also represent:
- Positive and negative zero (+0.0 and -0.0)
- Positive and negative infinity
- NaN (Not a Number)
*/
pub trait Float:
    Copy
    + Sum
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(n: usize) -> Option<Self>;
    fn powi(self, n: i32) -> Self;
    fn sqrt(self) -> Self;
}

/// Common code for floating point types.
macro_rules! impl_float {
    ($($t:ty),*) => {
        $(
            impl Float for $t {
                #[inline]
                fn zero() -> Self { 0.0 }
                #[inline]
                fn one() -> Self { 1.0 }
                #[inline]
                fn from_usize(n: usize) -> Option<Self> { Some(n as Self) }
                #[inline]
                fn powi(self, n: i32) -> Self { self.powi(n) }
                #[inline]
                fn sqrt(self) -> Self { self.sqrt() }
            }
        )*
    }
}

impl_float!(f32, f64);

/* ########################### Utility functions ########################### */

/// Sort a vector in place, based on the current and desired sorting state.
fn sort_vec<T: Ord>(v: &mut Vec<T>, state: &SortState, desired: Sorting) {
    // short circuit no-ops
    let len: usize = v.len();
    if len == 0 || len == 1 || desired == Sorting::None {
        return;
    } else if state == &SortState::Asc && desired == Sorting::Ascending {
        return;
    } else if state == &SortState::Desc && desired == Sorting::Descending {
        return;
    }

    match state {
        SortState::Changed | SortState::Unsorted => {
            if desired == Sorting::Ascending {
                v.sort();
            } else {
                v.sort_by(|a, b| b.cmp(a));
            }
        }
        SortState::Asc | SortState::Desc => {
            v.reverse();
        }
        SortState::SwappedToFront(num, prevstate) => {
            if **prevstate == SortState::Asc || **prevstate == SortState::Desc {
                // Restore the original 1st element back to the front.
                // Since the vec used to be sorted, this should make the
                // actual sorting algo's work a little easier.
                v.swap(0, len - *num);
            }
            // recurse back to this function
            sort_vec(v, &SortState::Changed, desired);
        }
    }
}

/* ######################################################################### */

#[cfg(test)]
mod tests {
    use super::*;

    const PI_LEN: usize = 16;
    const PI_SUM: u32 = 80;
    const PI_ARR: [u32; PI_LEN] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
    const PI_ASC: [u32; PI_LEN] = [1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9];
    const PI_DESC: [u32; PI_LEN] = [9, 9, 9, 8, 7, 6, 5, 5, 5, 4, 3, 3, 3, 2, 1, 1];
    const FP_ARR: [f64; 7] = [-999.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999.0];
    const XTRA: u32 = 99;
    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_new_and_push() {
        let mut ev1: EnhVec<u32> = EnhVec::new();
        assert!(ev1.is_empty());
        ev1.push(PI_ARR[0]);
        ev1.push(PI_ARR[1]);
        assert_eq!(ev1.len(), 2);
        assert_eq!(ev1[0], PI_ARR[0]);
        assert_eq!(ev1[1], PI_ARR[1]);

        let ev2: EnhVec<u32> = EnhVec::new();
        assert_ne!(ev1, ev2);
    }

    #[test]
    fn test_from_iter() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.len(), PI_LEN);
        assert_eq!(ev[0], PI_ARR[0]);
        assert_eq!(ev[7], PI_ARR[7]);
    }

    #[test]
    fn test_is_sorted() {
        let t: Vec<i32> = vec![-1, 0, 1, 2, 3, 4, 5, 99];
        let ev1: EnhVec<i32> = EnhVec::new_from(t.clone());
        assert!(ev1.is_sorted());

        let ev2: EnhVec<&i32> = EnhVec::from_iter(t.iter().rev());
        assert!(!ev2.is_sorted());
    }

    #[test]
    fn test_sort_asc_and_iter() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.sort(Sorting::Ascending);
        assert_eq!(ev.len(), PI_LEN);

        let test: Vec<u32> = Vec::from_iter(PI_ASC);
        assert_eq!(ev.to_vec(), test);
        ev.iter()
            .enumerate()
            .for_each(|(i, x)| assert_eq!(x, &test[i]));
    }

    #[test]
    fn test_sort_desc_and_iter() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.sort(Sorting::Descending);
        assert_eq!(ev.len(), PI_LEN);

        let test: Vec<u32> = Vec::from_iter(PI_DESC);
        assert_eq!(ev.to_vec(), test);
        ev.iter()
            .enumerate()
            .for_each(|(i, x)| assert_eq!(x, &test[i]));
    }

    #[test]
    fn test_push_front() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.push_front(XTRA);
        assert_eq!(ev.len(), PI_LEN + 1);

        let mut test: Vec<u32> = Vec::from_iter(PI_ARR);
        test.insert(0, XTRA);
        assert_eq!(ev.to_vec(), test);
    }

    #[test]
    fn test_push_swap_front() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.push_swap_front(XTRA);
        assert_eq!(ev.len(), PI_LEN + 1);

        let mut test: Vec<u32> = Vec::from_iter(PI_ARR);
        test.push(XTRA);
        test.swap(0, PI_LEN);
        assert_eq!(ev.to_vec(), test);
    }

    #[test]
    fn test_insert_sorted() {
        let x: i32 = XTRA as i32;
        let mut ev: EnhVec<i32> = EnhVec::from_iter(vec![-x, 1, 3, 5, x]);
        assert_eq!(ev.len(), 5);
        ev.insert_sorted(4);
        assert_eq!(ev.len(), 6);
        assert_eq!(ev.to_vec(), vec![-x, 1, 3, 4, 5, x]);
    }

    #[test]
    fn test_as_ref_vec() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        let test: Vec<&u32> = vec![
            &3, &1, &4, &1, &5, &9, &2, &6, &5, &3, &5, &8, &9, &7, &9, &3,
        ];
        assert_eq!(ev.as_ref_vec(), test);
    }

    #[test]
    fn test_for_each() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);

        let mut sum: u32 = 0;
        ev.for_each(|x: &u32| sum += x);
        assert_eq!(sum, PI_SUM, "sum of all digits");

        let mut sum_if: u32 = 0;
        ev.for_each_if(|x: &u32| sum_if += x, |x: &u32| x % 2 == 0);
        assert_eq!(sum_if, 20, "sum of even digits");
    }

    #[test]
    fn test_modify_each() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);

        ev.modify_each(|x: &mut u32| *x += 1);
        assert_eq!(ev.to_vec(), vec![4, 2, 5, 2, 6, 10, 3, 7, 6, 4, 6, 9, 10, 8, 10, 4]);

        ev.modify_each_if(|x: &mut u32| *x -= 1, |x: &u32| x > &5);
        assert_eq!(ev.to_vec(), vec![4, 2, 5, 2, 5, 9, 3, 6, 5, 4, 5, 8, 9, 7, 9, 4]);
    }

    #[test]
    fn test_count() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.count(&0), 0);
        assert_eq!(ev.count(&1), 2);
        assert_eq!(ev.count(&2), 1);
        assert_eq!(ev.count(&3), 3);
        assert_eq!(ev.count(&4), 1);
        assert_eq!(ev.count(&5), 3);
        assert_eq!(ev.count(&6), 1);
        assert_eq!(ev.count(&7), 1);
        assert_eq!(ev.count(&8), 1);
        assert_eq!(ev.count(&9), 3);
        assert_eq!(ev.count(&XTRA), 0);
    }

    #[test]
    fn test_sum() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.sum(), PI_SUM);
    }

    #[test]
    fn test_range() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.range(), Some(8));
    }

    #[test]
    fn test_median() {
        let ev1: EnhVec<i32> = EnhVec::from_iter(vec![1, 3, 5]);
        assert_eq!(ev1.median(), Some(3));

        let ev2: EnhVec<i32> = EnhVec::from_iter(vec![1, 2, 3, 4]);
        assert_eq!(ev2.median(), Some(2));
    }

    #[test]
    fn test_mode() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        // 3, 5 and 9 have the same count
        assert!(matches!(ev.mode(), Some(3) | Some(5) | Some(9)), "mode is not 3, 5 or 9");
    }

    #[test]
    fn test_average() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.average(), Some(5.0));
    }

    #[test]
    fn test_product() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        let mut prod: i128 = 1;
        ev.for_each(|x: &u32| prod *= *x as i128);
        assert_eq!(ev.product(), prod);
    }

    #[test]
    fn test_variance_and_stdev() {
        let ev: EnhVec<u32> = EnhVec::from_iter(vec![2, 4, 4, 4, 5, 5, 7, 9]);
        let var_diff: f64 = ev.variance().unwrap() - 4.0;
        let std_diff: f64 = ev.stdev().unwrap() - 2.0;
        assert!(var_diff.abs() < EPSILON, "variance diff ({var_diff}) not within epsilon");
        assert!(std_diff.abs() < EPSILON, "stdev diff ({std_diff}) not within epsilon");
    }

    #[test]
    fn test_distinct() {
        let x: i32 = XTRA as i32;
        let data: Vec<i32> = vec![x, 1, 2, 2, -x, 3, 3, 3, 4];
        let ev: EnhVec<i32> = EnhVec::from_iter(data);
        let distinct: EnhVec<i32> = ev.distinct(Some(Sorting::Ascending));
        assert_eq!(distinct.to_vec(), vec![-x, 1, 2, 3, 4, x]);
    }

    #[test]
    fn test_percentile() {
        let ev: EnhVec<u32> = EnhVec::from_iter(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, XTRA]);
        assert_eq!(ev.percentile(0.00), Some(0), "percentile 0.0");
        assert_eq!(ev.percentile(0.01), Some(0), "percentile 0.01");
        assert_eq!(ev.percentile(0.10), Some(1), "percentile 0.1");
        assert_eq!(ev.percentile(0.20), Some(2), "percentile 0.2");
        assert_eq!(ev.percentile(0.30), Some(3), "percentile 0.3");
        assert_eq!(ev.percentile(0.40), Some(4), "percentile 0.4");
        assert_eq!(ev.percentile(0.50), Some(5), "percentile 0.5");
        assert_eq!(ev.percentile(0.54), Some(5), "percentile 0.54");
        assert_eq!(ev.percentile(0.55), Some(6), "percentile 0.55");
        assert_eq!(ev.percentile(0.60), Some(6), "percentile 0.6");
        assert_eq!(ev.percentile(0.70), Some(7), "percentile 0.7");
        assert_eq!(ev.percentile(0.80), Some(8), "percentile 0.8");
        assert_eq!(ev.percentile(0.90), Some(9), "percentile 0.9");
        assert_eq!(ev.percentile(0.95), Some(XTRA), "percentile 0.95");
        assert_eq!(ev.percentile(1.00), Some(XTRA), "percentile 1.0");
    }

    #[test]
    fn test_median_fp() {
        let ev: EnhVec<f64> = EnhVec::from_iter(FP_ARR);
        let diff: f64 = ev.median_fp().unwrap() - 3.0;
        assert!(diff.abs() < EPSILON, "median diff ({diff}) not within epsilon");
    }

    #[test]
    fn test_average_fp() {
        let ev: EnhVec<f32> = EnhVec::from_iter(FP_ARR.iter().map(|&x| x as f32));
        let diff: f32 = ev.average_fp().unwrap() - 2.142857143; // 15 / 7
        assert!(diff.abs() < EPSILON as f32, "avg diff ({diff}) not within epsilon");
    }

    #[test]
    fn test_product_fp() {
        let ev: EnhVec<f64> = EnhVec::from_iter(FP_ARR);
        let mut prod: f64 = 1.0;
        ev.for_each(|x: &f64| prod *= x);
        let diff: f64 = ev.product_fp() - prod;
        assert!(diff.abs() < EPSILON, "product diff ({diff}) not within epsilon");
    }

    #[test]
    fn test_empty_vec() {
        let ev: EnhVec<i32> = EnhVec::new();
        assert!(ev.is_empty());
        assert_eq!(ev.sum(), 0, "sum is not zero");
        assert_eq!(ev.product(), 1, "product is not one");
        assert_eq!(ev.range(), None, "range is not None");
        assert_eq!(ev.median(), None, "median is not None");
        assert_eq!(ev.mode(), None, "mode is not None");
        assert_eq!(ev.average(), None, "average is not None");
        assert_eq!(ev.percentile(-1.), None, "percentile < 0 is not None");
        assert_eq!(ev.percentile(0.5), None, "percentile is not None");
        assert_eq!(ev.percentile(1.5), None, "percentile > 1 is not None");
        assert!(ev.is_sorted(), "is_sorted");
    }

    #[test]
    fn test_single_element() {
        let purpose: i32 = 42;
        let ev: EnhVec<i32> = EnhVec::from_iter(vec![purpose]);
        assert_eq!(ev.sum(), purpose, "sum");
        assert_eq!(ev.product(), purpose.into(), "product");
        assert_eq!(ev.range(), Some(0), "range");
        assert_eq!(ev.median(), Some(purpose), "median");
        assert_eq!(ev.mode(), Some(purpose), "mode");
        assert_eq!(ev.average(), Some(purpose as f64), "average");
        assert_eq!(ev.percentile(0.01), Some(purpose), "percentile 0.01");
        assert_eq!(ev.percentile(0.50), Some(purpose), "percentile 0.50");
        assert_eq!(ev.percentile(0.99), Some(purpose), "percentile 0.99");
        assert!(ev.is_sorted(), "is_sorted");
    }
}
