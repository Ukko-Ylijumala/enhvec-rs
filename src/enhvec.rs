// Copyright (c) 2024 Mikko Tanner. All rights reserved.

#![allow(dead_code)]

use crate::hashing::{CustomXxh3Hasher, Xxh3Hashable};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    iter::Sum,
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
};

/**
A wrapper around a Vec of elements (objects/items).

This struct provides additional methods for handling elements:
- sorting the elements ascending or descending
- returning references to the elements, also sorted
- pushing elements to the front of the vector
- hashing the elements in a stable, repeatable way
*/
#[derive(Debug, Default, Clone)]
pub struct EnhVec<T>(Vec<T>);

// Technically PartialEq and PartialOrd bounds are not needed for the
// methods in this block, but we want to restrict the types allowed
// in EhnVec to those that can be compared and sorted.
impl<T: PartialEq + PartialOrd> EnhVec<T> {
    pub fn new() -> Self {
        Self(Vec::<T>::new())
    }
    pub fn new_from(elements: Vec<T>) -> Self {
        Self(elements)
    }
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
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
        self.0.insert(0, element)
    }

    /**
    Insert an element at the end of the [EnhVec], then swap it with the
    first element. This is a much faster way to push an element to the
    front of the Vec than `push_front()`, as it doesn't require shifting
    all other elements. Time complexity: `O(1)`.
    */
    pub fn push_swap_front(&mut self, element: T) {
        let last: usize = self.len(); // len() - 1 after push()
        self.push(element);
        if last > 0 {
            self.swap(0, last);
        }
    }

    /// Return a [Vec] of references to entries.
    pub fn as_ref_vec(&self) -> Vec<&T> {
        self.0.iter().collect()
    }
    /// Return a [Vec] of mutable references to entries.
    pub fn as_mut_ref_vec(&mut self) -> Vec<&mut T> {
        self.0.iter_mut().collect()
    }

    /// Run a closure on each element.
    pub fn for_each(&self, f: impl Fn(&T)) {
        self.0.iter().for_each(f)
    }
    /// Run a closure on each element if the predicate is true.
    pub fn for_each_if(&self, f: impl Fn(&T), predicate: impl Fn(&T) -> bool) {
        self.0.iter().for_each(|elem| {
            if predicate(elem) {
                f(elem)
            }
        })
    }

    /// Run a mutating closure for each element.
    pub fn modify_each(&mut self, f: impl FnMut(&mut T)) {
        self.0.iter_mut().for_each(f)
    }
    /// Run a mutating closure for each element if the predicate is true.
    pub fn modify_each_if(&mut self, mut f: impl FnMut(&mut T), predicate: impl Fn(&T) -> bool) {
        self.0.iter_mut().for_each(|elem| {
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
        self.0.clone()
    }
    /// Consume the [EnhVec] and return the inner [Vec<T>].
    pub fn into_vec(self) -> Vec<T> {
        self.0
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
        if (self.first().unwrap()).gt(&self.last().unwrap()) {
            // short circuit if first > last
            return false;
        }
        self.0
            .iter()
            .zip(self.0.iter().skip(1))
            .all(|(a, b)| a <= b)

        // TODO: check if this is faster than the iter().skip(1)
        // above for large Vecs and optimize accordingly
        //     self.0.windows(2).all(|w| w[0] <= w[1])
    }

    /**
    Insert an element into the [EnhVec] in sorted order.

    NOTE: the Vec must be sorted ASC for this insert to make much sense.
    If the Vec is not sorted, the insertion point would be more or less
    random, hence in this case we just `push()` the element to the end.

    NOTE: this method is potentially slow, as it traverses the Vec 2 times:
    once to check if it is sorted (worst case: `O(N)`), and once to find the
    insertion point (`O(log n)`). This may be be optimized in the future.

    NOTE: if you need to insert many elements, it will likely be faster to sort
    the Vec after all the insertions are done, as sorting is approx. `O(N log N)`.
    */
    pub fn insert_sorted(&mut self, element: T) {
        if self.is_empty() || element >= *self.last().unwrap() || !self.is_sorted() {
            // short circuit some common cases
            self.push(element);
            return;
        }

        let cutoff: usize = 20;
        if self.len() < cutoff {
            // linear search for "small" vectors
            let idx: usize = self.iter().position(|x| element < *x).unwrap_or(self.len());
            self.insert(idx, element);
        } else {
            // binary search for larger vectors
            let idx: usize = match self.binary_search(&element) {
                Ok(index) | Err(index) => index,
            };
            self.insert(idx, element);
        }
    }

    /// Normal sort: ascending order.
    pub fn sort_asc(&mut self) {
        self.0.sort()
    }
    /// Reverse sort: descending order.
    pub fn sort_desc(&mut self) {
        self.0.sort_by(|a, b| b.cmp(a))
    }

    /// Return references to entries in ASCending order.
    pub fn as_sorted_asc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        vec.sort();
        vec
    }
    /// Return references to entries in DESCending order.
    pub fn as_sorted_desc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        vec.sort_by(|a, b| b.cmp(a));
        vec
    }
}

/* --------------------------------- */

impl<T> Deref for EnhVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for EnhVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: PartialEq> EnhVec<T> {
    /// Count the occurrences of a value.
    pub fn count(&self, value: &T) -> usize {
        self.iter().filter(|&x| x == value).count()
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
    */
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_sorted_asc()
            .iter()
            .for_each(|elem| elem.hash(state));
    }
}

impl<T: Ord + Xxh3Hashable> Xxh3Hashable for EnhVec<T> {
    #[inline]
    fn xxh3<H: Hasher>(&self, state: &mut H) {
        self.as_sorted_asc()
            .iter()
            .for_each(|elem| elem.xxh3(state));
    }

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

        let mut counts: HashMap<T, i32> = HashMap::new();
        for &item in self.iter() {
            *counts.entry(item).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(item, _)| item)
    }

    /// Return the distinct (unique) elements, optionally sorted.
    pub fn distinct(&self, sorted: bool) -> EnhVec<T>
    where
        T: Copy + Eq + Hash + Ord,
    {
        let set: HashSet<T> = self.iter().copied().collect();
        let mut result: EnhVec<T> = EnhVec::from_iter(set.into_iter());
        if sorted {
            result.sort_asc();
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
        Some((sum / self.len() as i128) as f64)
    }

    /// Return the product of all elements.
    pub fn product(&self) -> T {
        self.iter().fold(T::one(), |acc, &x| acc * x)
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
        self.variance().map(|v| v.sqrt())
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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

/// In contrast to [Integer], we must remove [Eq] and [Ord] constraints, as
/// they are not defined for floating point numbers due to `NaN`. Also [Hash]
/// is not implemented for f32/f64, so we must remove that constraint as well.
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

/// Calculate the power of a number using the exponentiation by squaring method.
#[inline]
pub fn powi<T>(b: T, n: i64) -> f64
where
    T: Into<f64>,
{
    let b: f64 = b.into();
    if n == 0 {
        return 1.0;
    }
    let mut result: f64 = 1.0;
    let mut base: f64 = b;
    let mut exp: i64 = n.abs();

    while exp > 0 {
        if exp % 2 == 1 {
            result *= base;
        }
        exp /= 2;
        base *= base;
    }

    if n < 0 {
        1.0 / result
    } else {
        result
    }
}

/// Calculate the square root of a number using the Babylonian method.
#[inline]
pub fn sqrt<T>(n: T) -> f64
where
    T: Into<f64>,
{
    let n: f64 = n.into();
    if n < 0.0 {
        f64::NAN
    } else if n == 0.0 {
        0.0
    } else {
        let mut x: f64 = n;
        let mut y: f64 = 1.0;
        let e: f64 = 0.00000001; // precision
        while (x - y).abs() > e {
            x = (x + y) / 2.0;
            y = n / x;
        }
        x
    }
}
