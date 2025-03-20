// Copyright (c) 2024-2025 Mikko Tanner. All rights reserved.

use custom_xxh3::{CustomXxh3Hasher, Xxh3Hashable};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    iter::{Chain, Rev, Sum},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    slice::{Iter, IterMut},
};

/// The default size cutoff for sorting small vectors.
const SORT_SIZE_CUTOFF: usize = 20;
const HEAD_SIZE: usize = 16;

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
/// The expected sorting state of an [EnhVec].
pub enum Sorting {
    #[default]
    None,
    Ascending,
    Descending,
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
/// The current (internal) sorting state of an [EnhVec].
enum SortState {
    #[default]
    Unsorted,
    Asc,
    Desc,
    Changed, // changed - could be sorted or not, depending on what happened
}

impl SortState {
    #[inline]
    fn is_sorted(&self) -> bool {
        matches!(self, SortState::Asc | SortState::Desc)
    }

    #[inline]
    fn is_unsorted(&self) -> bool {
        matches!(self, SortState::Unsorted | SortState::Changed)
    }

    fn reverse(&mut self) {
        match self {
            SortState::Asc => *self = SortState::Desc,
            SortState::Desc => *self = SortState::Asc,
            _ => {}
        }
    }
}

/* --------------------------------- */

/// The actual internal representation of the [EnhVec].
#[derive(Debug, Default, Clone)]
struct EnhVecInner<T> {
    state: SortState,
    head: Vec<T>,
    main: Vec<T>,
}

impl<T> EnhVecInner<T> {
    fn new() -> Self {
        Self {
            state: SortState::Unsorted,
            head: Vec::with_capacity(HEAD_SIZE),
            main: Vec::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            main: Vec::with_capacity(capacity),
            ..Self::new()
        }
    }

    fn len(&self) -> usize {
        self.main.len() + self.head.len()
    }

    fn is_empty(&self) -> bool {
        self.head.is_empty() && self.main.is_empty()
    }

    fn first(&self) -> Option<&T> {
        self.head.last().or_else(|| self.main.first())
    }
    fn last(&self) -> Option<&T> {
        self.main.last().or_else(|| self.head.first())
    }

    /// Set the internal sorting state to "changed" if it isn't already.
    #[inline]
    fn set_changed(&mut self) {
        if self.state != SortState::Changed {
            self.state = SortState::Changed;
        }
    }

    fn insert(&mut self, idx: usize, element: T) {
        self.main.insert(idx, element);
        self.set_changed();
    }

    fn push(&mut self, element: T) {
        self.main.push(element);
        self.set_changed();
    }

    fn push_front(&mut self, element: T) {
        if self.head.len() + 1 > HEAD_SIZE {
            self.compact();
        }
        self.head.push(element);
    }

    fn push_swap_front(&mut self, element: T) {
        let last: usize = self.main.len(); // len() - 1 after push()
        self.main.push(element);
        if last > 0 {
            self.main.swap(0, last);
        }
        self.set_changed();
    }

    fn pop(&mut self) -> Option<T> {
        self.main.pop().or_else(|| {
            if self.head.is_empty() {
                None
            } else {
                Some(self.head.remove(0))
            }
        })
    }

    fn pop_front(&mut self) -> Option<T> {
        self.head.pop().or_else(|| {
            if self.main.is_empty() {
                None
            } else {
                if self.state.is_unsorted() {
                    // if the main Vec is unsorted, we can just swap-remove
                    return Some(self.main.swap_remove(0));
                }
                // removing the first element of a sorted Vec does not
                // change the ordering, so we can just remove it
                Some(self.main.remove(0))
            }
        })
    }

    /// Reverse the order of the elements in place and set state accordingly.
    fn reverse(&mut self) {
        self.head.reverse();
        self.main.reverse();
        if self.state.is_sorted() {
            self.state.reverse();
        } else {
            self.set_changed();
        }
    }

    /**
    Fold the head elements into the main Vec as the first K elements.

    Tries to minimize complexity by not reallocating. Instead the head elements
    are appended to the end of the main Vec, then rotated to the front. This is
    a "best effort" method, and may not always be most efficient.
    */
    fn compact(&mut self) {
        let k: usize = self.head.len();
        if k == 0 {
            return;
        }
        match self.state {
            SortState::Asc => {
                self.main.extend(self.head.drain(..).rev());
                self.main.rotate_right(k);
            }
            SortState::Desc => {
                self.main.extend(self.head.drain(..));
            }
            _ => {
                self.main.append(&mut self.head);
                self.set_changed();
            }
        }
    }

    // Internal iterators combining the head and main [Vec]s.
    fn internal_iter(&self) -> Chain<Rev<Iter<T>>, Iter<T>> {
        self.head.iter().rev().chain(self.main.iter())
    }
    fn internal_iter_mut(&mut self) -> Chain<Rev<IterMut<T>>, IterMut<T>> {
        self.set_changed(); // order of elements could change
        self.head.iter_mut().rev().chain(self.main.iter_mut())
    }
}

/* --------------------------------- */

// Allow indexing into EnhVecInner
impl<T> Index<usize> for EnhVecInner<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let head_len: usize = self.head.len();
        if index < head_len {
            // head elements are in reverse order -> reverse the index
            &self.head[head_len - 1 - index]
        } else {
            &self.main[index - head_len]
        }
    }
}

// Allow mutable indexing into EnhVecInner
impl<T> IndexMut<usize> for EnhVecInner<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // mutation could change the sort order of elements
        self.set_changed();
        let head_len: usize = self.head.len();
        if index < head_len {
            &mut self.head[head_len - 1 - index]
        } else {
            &mut self.main[index - head_len]
        }
    }
}

/* --------------------------------- */

impl<T: PartialOrd + Ord> EnhVecInner<T> {
    fn sort(&mut self, sorting: &Sorting) {
        sort_vec(&mut self.main, &self.state, sorting);
        self.state = match sorting {
            Sorting::Ascending => SortState::Asc,
            Sorting::Descending => SortState::Desc,
            _ => SortState::Unsorted,
        };
    }
}

impl<T: Ord> EnhVecInner<T> {
    /// Whether the [EnhVecInner] data is sorted in ascending order.
    fn is_sorted(&self) -> bool {
        match self.main.len() {
            0 | 1 => return true,
            2 => return self.main[0] <= self.main[1],
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
        self.main
            .iter()
            .zip(self.main.iter().skip(1))
            .all(|(a, b)| a <= b)

        // TODO: check if this is faster than the iter().skip(1)
        // above for large Vecs and optimize accordingly
        // must check for empty first to avoid panic with `windows()` method
        //     self.v.windows(2).all(|w| w[0] <= w[1])
    }

    fn insert_sorted(&mut self, element: T) {
        if self.is_empty() || element >= *self.main.last().unwrap() {
            // short circuit some common cases
            self.main.push(element);
            return;
        }

        // determine the insertion point
        self.compact();
        let idx: usize = match self.main.len() < SORT_SIZE_CUTOFF {
            // linear search for "small" vectors
            true => self
                .internal_iter()
                .position(|x: &T| element < *x)
                .unwrap_or(self.main.len()),
            // binary search for larger vectors
            false => match self.main.binary_search(&element) {
                Ok(index) | Err(index) => index,
            },
        };
        self.main.insert(idx, element);
    }
}

/* --------------------------------- */

impl<T: PartialEq> PartialEq for EnhVecInner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.main == other.main
    }
}

impl<T> From<Vec<T>> for EnhVecInner<T> {
    fn from(v: Vec<T>) -> Self {
        Self {
            main: v,
            ..Self::new()
        }
    }
}

/* ######################## Main EnhVec structure ######################## */

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
    data: EnhVecInner<T>,
    sort: Sorting,
}

// Technically PartialEq and PartialOrd bounds are not needed for the
// methods in this block, but we want to restrict the types allowed
// in EnhVec to those that can be compared and sorted.
impl<T: PartialEq + PartialOrd> EnhVec<T> {
    fn default() -> Self {
        Self {
            data: EnhVecInner::new(),
            sort: Sorting::None,
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
            data: EnhVecInner::with_capacity(capacity),
            ..Self::default()
        }
    }
    pub fn new_from(elements: Vec<T>) -> Self {
        Self {
            data: elements.into(),
            ..Self::default()
        }
    }
    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect::<Vec<T>>().into(),
            ..Self::default()
        }
    }

    /// Insert an element at index. Possibly slow, as it may shift other elements.
    pub fn insert(&mut self, idx: usize, element: T) {
        self.data.insert(idx, element);
    }

    /// Push an element to the end of the [EnhVec].
    pub fn push(&mut self, element: T) {
        self.data.push(element);
    }

    /**
    Insert an element at the start of the [EnhVec].

    NOTE: potentially slow, as it may have to fold the head elements into
    the main Vec. Time complexity: `O(N)` in that case. Prefer `push()`
    and finally `sort()` if you need to maintain a certain order, or
    `push_swap_front()` if you just need the new element to be the
    first one and don't particularly care about the rest.
    */
    pub fn push_front(&mut self, element: T) {
        self.data.push_front(element);
    }

    /**
    Insert an element at the end of the [EnhVec], then swap it with the
    first element. This is a much faster way to push an element to the
    front of the Vec than `push_front()`, as it doesn't require shifting
    all other elements. Time complexity: `O(1)`.
    */
    pub fn push_swap_front(&mut self, element: T) {
        self.data.push_swap_front(element);
    }
}

/* --------------------------------- */

// Generic methods for all types
impl<T> EnhVec<T> {
    /// Reverse the order of the elements in place. [Sorting] is updated.
    pub fn reverse(&mut self) {
        self.data.reverse();
        match self.data.state {
            SortState::Changed | SortState::Unsorted => {
                self.sort = Sorting::None;
            }
            SortState::Asc => {
                self.sort = Sorting::Ascending;
            }
            SortState::Desc => {
                self.sort = Sorting::Descending;
            }
        }
    }

    /// Return a [Vec] of references to entries.
    pub fn as_ref_vec(&self) -> Vec<&T> {
        self.data.internal_iter().collect()
    }
    /// Return a [Vec] of mutable references to entries.
    pub fn as_mut_ref_vec(&mut self) -> Vec<&mut T> {
        self.data.internal_iter_mut().collect()
    }

    /// Run a closure on each element.
    pub fn for_each(&self, f: impl FnMut(&T)) {
        self.data.internal_iter().for_each(f)
    }
    /// Run a closure on each element if the predicate is true.
    pub fn for_each_if(&self, mut f: impl FnMut(&T), predicate: impl Fn(&T) -> bool) {
        self.data.internal_iter().for_each(|elem: &T| {
            if predicate(elem) {
                f(elem)
            }
        })
    }

    /// Run a mutating closure for each element.
    pub fn modify_each(&mut self, f: impl FnMut(&mut T)) {
        self.data.internal_iter_mut().for_each(f)
    }
    /// Run a mutating closure for each element if the predicate is true.
    pub fn modify_each_if(&mut self, mut f: impl FnMut(&mut T), predicate: impl Fn(&T) -> bool) {
        self.data.internal_iter_mut().for_each(|elem: &mut T| {
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
        let mut data: Vec<T> = self.data.head.clone();
        data.reverse();
        data.extend(self.data.main.clone());
        data
    }

    /// Consume the [EnhVec] and return the inner [Vec<T>].
    pub fn into_vec(self) -> Vec<T> {
        self.data
            .head
            .into_iter()
            .rev()
            .chain(self.data.main.into_iter())
            .collect()
    }

    /// Length of the [EnhVec] (the sum of the lengths of the head and main [Vec]s).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the [EnhVec] is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn first(&self) -> Option<&T> {
        self.data.first()
    }
    pub fn last(&self) -> Option<&T> {
        self.data.last()
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    // TODO: find a way to have this return a "normal" Iter<T> and not a Chain<...>
    pub fn iter(&self) -> Chain<Rev<Iter<T>>, Iter<T>> {
        self.data.internal_iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<T> {
        // FIXME: implement a proper iter_mut() method
        self.data.main.iter_mut()
    }
}

/* --------------------------------- */

impl<T: Ord> EnhVec<T> {
    /**
    Whether the [EnhVec] is sorted in ascending order. Worst case time
    complexity: `O(N)`, as it may have to compare each element with the next.

    NOTE: an empty or 1-element EnhVec is considered sorted.
    */
    pub fn is_sorted(&self) -> bool {
        self.data.is_sorted()
    }

    /**
    Insert an element into the [EnhVec] in sorted order.

    NOTE: data must be sorted ASC or DESC for this insert to make much sense.
    If the data is not sorted, the insertion point would be more or less
    random, hence in this case we just `push()` the element to the end.

    NOTE: this method is potentially slow, as it might traverse the data twice:
    once to check if it is sorted (worst case: `O(N)`), and once to find the
    insertion point (`O(log n)`). This may be be optimized in the future.

    NOTE: if you need to add many elements, it will likely be faster to push()
    and finally sort() after all the insertions are done, as sorting is approx.
    `O(N log N)`.
    */
    pub fn insert_sorted(&mut self, element: T) {
        if !self.is_sorted() {
            self.push(element);
            return;
        }
        self.data.insert_sorted(element);
    }

    /// Set the default sorting state of the [EnhVec] and sort the data.
    pub fn sort(&mut self, sorting: Sorting) {
        if sorting == self.sort {
            return;
        }
        self.data.sort(&sorting);
        self.sort = sorting;
    }

    /// Return references to entries in ASCending order.
    pub fn as_sorted_asc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        sort_vec(&mut vec, &self.data.state, &Sorting::Ascending);
        vec
    }
    /// Return references to entries in DESCending order.
    pub fn as_sorted_desc(&self) -> Vec<&T> {
        let mut vec: Vec<&T> = self.as_ref_vec();
        sort_vec(&mut vec, &self.data.state, &Sorting::Descending);
        vec
    }
}

/* --------------------------------- */

impl<T: PartialEq> EnhVec<T> {
    /// Count the occurrences of a value.
    pub fn count(&self, value: &T) -> usize {
        self.data.internal_iter().filter(|&x| x == value).count()
    }

    /// Check if the [EnhVec] contains a value.
    pub fn contains(&self, value: &T) -> bool {
        self.data.internal_iter().any(|x: &T| x == value)
    }
    /// Check if the [EnhVec] contains all values in another [EnhVec].
    pub fn contains_all(&self, other: &Self) -> bool {
        other.data.internal_iter().all(|x: &T| self.contains(x))
    }
    /// Check if the [EnhVec] contains any values in another [EnhVec].
    pub fn contains_any(&self, other: &Self) -> bool {
        other.data.internal_iter().any(|x: &T| self.contains(x))
    }
    /// Check if the [EnhVec] contains only values in another [EnhVec].
    pub fn contains_only(&self, other: &Self) -> bool {
        self.contains_all(other) && self.len() == other.len()
    }

    /// Check if the [EnhVec] is a subset of another [EnhVec].
    pub fn is_subset(&self, other: &Self) -> bool {
        self.contains_all(other)
    }
    /// Check if the [EnhVec] is a superset of another [EnhVec].
    pub fn is_superset(&self, other: &Self) -> bool {
        self.contains_any(other)
    }
    /// Check if the [EnhVec] is disjoint with another [EnhVec].
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.contains_any(other)
    }
    /// Check if the [EnhVec] is equal to another [EnhVec].
    pub fn is_equal(&self, other: &Self) -> bool {
        self.contains_only(other)
    }

    /// Check if the [EnhVec] is a proper subset of another [EnhVec].
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self.is_subset(other) && self.len() < other.len()
    }

    /// Check if the [EnhVec] is a proper superset of another [EnhVec].
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self.is_superset(other) && self.len() > other.len()
    }
    /// Check if the [EnhVec] is a proper subset or superset of another [EnhVec].
    pub fn is_proper(&self, other: &Self) -> bool {
        self.is_proper_subset(other) || self.is_proper_superset(other)
    }
    /// Check if the [EnhVec] is a proper subset and superset of another [EnhVec].
    pub fn is_proper_both(&self, other: &Self) -> bool {
        self.is_proper_subset(other) && self.is_proper_superset(other)
    }
    /// Check if the [EnhVec] is a proper subset or superset of another [EnhVec].
    pub fn is_proper_either(&self, other: &Self) -> bool {
        self.is_proper_subset(other) || self.is_proper_superset(other)
    }
}

/* --------------------------------- */

impl<T: PartialEq> PartialEq for EnhVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

// Allow indexing into EnhVec
impl<T> Index<usize> for EnhVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

// Allow mutable indexing into EnhVec
impl<T> IndexMut<usize> for EnhVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: PartialEq + PartialOrd> From<Vec<T>> for EnhVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self::new_from(v)
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

impl<T: Copy + Sum> EnhVec<T> {
    /// Return the sum of all elements.
    pub fn sum(&self) -> T {
        self.data.internal_iter().copied().sum()
    }
}

impl<T> EnhVec<T>
where
    T: Copy + Ord + Sub<Output = T>,
{
    /// Return the range (max - min) of the elements.
    pub fn range(&self) -> Option<T> {
        if let (Some(min), Some(max)) = (
            self.data.internal_iter().min(),
            self.data.internal_iter().max(),
        ) {
            Some(*max - *min)
        } else {
            None
        }
    }
}

/* --------------------------------- */

impl<T: Copy + Eq + Hash> EnhVec<T> {
    /// Return the mode (most common) value of the elements.
    pub fn mode(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut counts: HashMap<T, u32> = HashMap::new();
        for &item in self.data.internal_iter() {
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
        let set: HashSet<T> = self.data.internal_iter().copied().collect();
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

        let sum: i128 = self.data.internal_iter().copied().sum::<T>().into();
        Some(sum as f64 / self.len() as f64)
    }

    /// Return the product of all elements. For empty EnhVec, `product == 1`.
    /// To avoid overflow, multiplications are performed as `i128`.
    pub fn product(&self) -> i128
    where
        T: Into<i128>,
    {
        self.data
            .internal_iter()
            .fold(1, |acc: i128, &x| acc * x.into())
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
            .data
            .internal_iter()
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

        let sum: T = self.data.internal_iter().copied().sum();
        Some(sum / T::from_usize(self.len()).unwrap())
    }

    /// Return the product of all elements. Floating point version.
    pub fn product_fp(&self) -> T {
        self.data.internal_iter().fold(T::one(), |acc, &x| acc * x)
    }

    /// Return the variance of the elements. Floating point version.
    pub fn variance_fp(&self) -> Option<T> {
        if self.len() < 2 {
            return None;
        }

        let mean: T = self.average_fp()?;
        let variance: T = self
            .data
            .internal_iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<T>()
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
fn sort_vec<T: Ord>(v: &mut Vec<T>, state: &SortState, desired: &Sorting) {
    // short circuit no-ops
    if matches!(v.len(), 0 | 1) || *desired == Sorting::None {
        return;
    } else if *state == SortState::Asc && *desired == Sorting::Ascending {
        return;
    } else if *state == SortState::Desc && *desired == Sorting::Descending {
        return;
    }

    if state.is_unsorted() {
        if *desired == Sorting::Ascending {
            v.sort();
        } else {
            v.sort_by(|a, b| b.cmp(a));
        }
    } else {
        // we already know the vec is sorted, but not in the desired order
        // (because we checked for that in the short circuit above),
        // so we can just reverse it to get the other order
        v.reverse();
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
        ev.data
            .internal_iter()
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
        ev.data
            .internal_iter()
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
        let test: Vec<&u32> = PI_ARR.iter().collect();
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
    #[rustfmt::skip]
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
    #[rustfmt::skip]
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
    #[rustfmt::skip]
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
    #[rustfmt::skip]
    fn test_median_fp() {
        let ev: EnhVec<f64> = EnhVec::from_iter(FP_ARR);
        let diff: f64 = ev.median_fp().unwrap() - 3.0;
        assert!(diff.abs() < EPSILON, "median diff ({diff}) not within epsilon");
    }

    #[test]
    #[rustfmt::skip]
    fn test_average_fp() {
        let ev: EnhVec<f32> = EnhVec::from_iter(FP_ARR.iter().map(|&x| x as f32));
        let diff: f32 = ev.average_fp().unwrap() - 2.142857143; // 15 / 7
        assert!(diff.abs() < EPSILON as f32, "avg diff ({diff}) not within epsilon");
    }

    #[test]
    #[rustfmt::skip]
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
