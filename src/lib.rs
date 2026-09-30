// Copyright (c) 2024-2026 Mikko Tanner. All rights reserved.

use custom_xxh3::{CustomXxh3Hasher, Xxh3Hashable};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    iter::{Chain, FusedIterator, Rev, Sum},
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    slice::{Iter, IterMut},
};

/// The default size cutoff for linear/binary search.
const SEARCH_SIZE_CUTOFF: usize = 32;
/// Head size beyond which `push_swap_front()` swaps elements around instead of just pushing.
const HEAD_SIZE: usize = 16;
/**
Size (in bytes) of a main Vec from which growing it in place beats moving it
into a fresh allocation, see `benches/vec_insert.rs`. Above roughly this size
allocators (e.g. glibc) typically remap pages on realloc instead of copying.
*/
const LARGE_VEC_BYTES: usize = 128 * 1024;

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
#[derive(Debug, Clone)]
struct EnhVecInner<T> {
    state: SortState,
    head: Vec<T>,
    main: Vec<T>,
}

impl<T> EnhVecInner<T> {
    fn sort_by<F>(&mut self, f: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.set_changed(); // cannot know what `f` does to the order
        self.compact();
        self.main.sort_by(f);
    }

    fn new() -> Self {
        Self {
            state: SortState::Unsorted,
            head: Vec::new(),
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

    /// Set the internal sorting state to "changed".
    #[inline]
    fn set_changed(&mut self) {
        self.state = SortState::Changed;
    }

    /**
    Constant time push to the front of [EnhVecInner]. This method attempts
    to maintain a semblance of order by swapping elements within the head
    and between the head and main Vecs. Specifically, it moves the first
    element of the head to the main Vec, adds the new element to the head,
    and performs swaps to keep the head partially ordered.
    */
    fn push_swap_front(&mut self, element: T) {
        let head_len: usize = self.head.len();
        if head_len < HEAD_SIZE {
            // if the head Vec is not full, just push to it
            self.head.push(element);
        } else {
            // move the 0th element of the head Vec to the main Vec
            // and push the new element to the head Vec
            let main_last: usize = self.main.len(); // len() - 1 after push()
            self.main.push(self.head.swap_remove(0));
            self.head.push(element);
            self.head.swap(0, head_len - 2);
            if main_last > 0 {
                self.main.swap(0, main_last);
            }
        }
        self.set_changed();
    }

    /**
    Pop from the back of main. If main is empty, the back half of the head
    is moved over first, so that any mix of pops from both ends stays
    amortized `O(1)` (each move is paid for by the pops it enables).
    */
    fn pop(&mut self) -> Option<T> {
        if self.main.is_empty() && !self.head.is_empty() {
            // head[..k] are the last k elements, stored reversed
            let k: usize = self.head.len().div_ceil(2);
            self.main.extend(self.head.drain(..k).rev());
        }
        self.main.pop()
    }

    /// Pop from the front, i.e. the end of the head. If the head is empty, the
    /// front half of main is moved over first, like in `pop()`.
    fn pop_front(&mut self) -> Option<T> {
        if self.head.is_empty() && !self.main.is_empty() {
            let k: usize = self.main.len().div_ceil(2);
            self.head.extend(self.main.drain(..k).rev());
        }
        self.head.pop()
    }

    /// Like `pop_front()`, but swap-removes from an unsorted main Vec.
    fn swap_pop_front(&mut self) -> Option<T> {
        if self.head.is_empty() && !self.main.is_empty() && self.state.is_unsorted() {
            // no known order to maintain, so we can just swap-remove
            return Some(self.main.swap_remove(0));
        }
        // removing the first element of a sorted Vec does not
        // change the ordering, so we can just remove it
        self.pop_front()
    }

    /**
    Reverse the order of the elements in place and set state accordingly.

    The logical order is `rev(head) ++ main`, so the reverse is `rev(main) ++ head`.
    The head is already stored in the order the new tail needs, so it is
    simply appended to the reversed main Vec.
    */
    fn reverse(&mut self) {
        self.main.reverse();
        self.main.append(&mut self.head);
        if self.state.is_sorted() {
            self.state.reverse();
        } else {
            self.set_changed();
        }
    }

    /**
    Fold the head elements into the main Vec. The order of the elements is
    preserved. If the head is empty, this is a no-op.

    Tries to minimize computational complexity (see `benches/vec_insert.rs`):
    - with enough spare capacity, main is shifted once, in place
    - else for small main, a new Vec is allocated and elements are moved to it,
      as growing in place would copy main twice (realloc + shift)
    - else main is grown and shifted in place, as large reallocations are cheap

    Only as much capacity is added as needed (the existing one is kept), as
    compaction happens before sorting, `into_vec()` etc., not on pushes.
    */
    fn compact(&mut self) {
        let head_len: usize = self.head.len();
        if head_len == 0 {
            return;
        }

        let main_len: usize = self.main.len();
        let fits: bool = self.main.capacity() - main_len >= head_len;
        if fits || main_len * size_of::<T>() >= LARGE_VEC_BYTES {
            // splice() would otherwise grow like Vec::reserve(), up to doubling
            self.main.reserve_exact(head_len);
            self.main.splice(0..0, self.head.drain(..).rev());
        } else {
            let capacity: usize = (main_len + head_len).max(self.main.capacity());
            let mut tmp: Vec<T> = Vec::with_capacity(capacity);
            tmp.extend(self.head.drain(..).rev());
            tmp.append(&mut self.main);
            self.main = tmp;
        }
        if self.state.is_unsorted() {
            self.set_changed();
        }
    }

    // Internal iterators combining the head and main [Vec]s.
    fn internal_iter(&'_ self) -> Chain<Rev<Iter<'_, T>>, Iter<'_, T>> {
        self.head.iter().rev().chain(self.main.iter())
    }
    fn internal_iter_mut(&'_ mut self) -> Chain<Rev<IterMut<'_, T>>, IterMut<'_, T>> {
        self.set_changed(); // order of elements could change
        self.head.iter_mut().rev().chain(self.main.iter_mut())
    }

    /**
    Whether logical `index` is in the head, and its position in that Vec.
    Written so that the Vec is picked with a conditional move instead of a
    jump, which the CPU would mispredict when reads hit head and main at
    random. An out of bounds position is caught by the slice indexing.
    */
    #[inline]
    fn locate(&self, index: usize) -> (bool, usize) {
        let head_len: usize = self.head.len();
        let in_head: bool = index < head_len;
        // head elements are in reverse order -> reverse the index
        let head_pos: usize = head_len.wrapping_sub(1).wrapping_sub(index);
        let main_pos: usize = index.wrapping_sub(head_len);
        (in_head, if in_head { head_pos } else { main_pos })
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            Some(&self[index])
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            None
        } else {
            Some(&mut self[index])
        }
    }
}

/* --------------------------------- */

// Allow indexing into EnhVecInner
impl<T> Index<usize> for EnhVecInner<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let (in_head, i): (bool, usize) = self.locate(index);
        let slice: &[T] = if in_head { &self.head } else { &self.main };
        &slice[i]
    }
}

// Allow mutable indexing into EnhVecInner
impl<T> IndexMut<usize> for EnhVecInner<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // mutation could change the sort order of elements
        self.set_changed();
        let (in_head, i): (bool, usize) = self.locate(index);
        let slice: &mut [T] = if in_head {
            &mut self.head
        } else {
            &mut self.main
        };
        &mut slice[i]
    }
}

/* --------------------------------- */

impl<T: PartialOrd> EnhVecInner<T> {
    /// Whether the [EnhVecInner] data is sorted in ascending order.
    fn is_sorted(&self) -> bool {
        self.is_sorted_in(SortState::Asc)
    }

    /// Whether the [EnhVecInner] data is sorted in descending order.
    fn is_sorted_desc(&self) -> bool {
        self.is_sorted_in(SortState::Desc)
    }

    /// Whether the data is sorted in the given `order` (ASC or DESC).
    fn is_sorted_in(&self, order: SortState) -> bool {
        // whether `a` may precede `b` in the requested order
        let desc: bool = order == SortState::Desc;
        let in_order = |a: &T, b: &T| if desc { a >= b } else { a <= b };
        match self.len() {
            0 | 1 => return true,
            2 => return in_order(&self[0], &self[1]),
            _ => {}
        }
        if self.state == order {
            // short circuit if already sorted
            return true;
        }
        if !in_order(self.first().unwrap(), self.last().unwrap()) {
            // short circuit if first and last are out of order
            return false;
        }

        /*
        Check the slices directly, which is several times faster than zipping
        the chained iterators, with the direction decided outside the loops so
        they can be vectorized. The head is stored reversed.
        */
        let junction: bool = match (self.head.first(), self.main.first()) {
            (Some(h), Some(m)) => in_order(h, m),
            _ => true,
        };
        let (head, main): (&[T], &[T]) = (&self.head, &self.main);
        junction
            && match desc {
                false => head.is_sorted_by(|a: &T, b: &T| a >= b) && main.is_sorted(),
                true => head.is_sorted() && main.is_sorted_by(|a: &T, b: &T| a >= b),
            }
    }

    /// Inserts into the head or main Vec, depending on the index.
    fn insert(&mut self, idx: usize, element: T) {
        let head_len: usize = self.head.len();
        if idx < head_len {
            /*
            insert into the head Vec even if it's "full", since this likely
            saves some extra work now and we can always compact it later.
            Head is reversed: position `head_len - idx` ends up at `idx`.
            */
            self.head.insert(head_len - idx, element);
        } else {
            // insert into the main Vec
            self.main.insert(idx - head_len, element);
        }
        if self.order_changed(idx) {
            self.set_changed();
        }
    }

    fn push(&mut self, element: T) {
        self.main.push(element);
        // nothing to check once the order is known to have changed
        if self.state != SortState::Changed && self.order_changed(self.len() - 1) {
            self.set_changed();
        }
    }

    fn push_front(&mut self, element: T) {
        // the head is a reversed Vec, so this is a plain (amortized O(1)) push
        self.head.push(element);
        if self.state != SortState::Changed && self.order_changed(0) {
            self.set_changed();
        }
    }

    /// Check whether an element addition changed the sort ordering.
    #[inline]
    fn order_changed(&self, idx: usize) -> bool {
        if self.len() == 1 {
            return false;
        }

        match self.state {
            SortState::Unsorted => true,
            SortState::Changed => false,
            SortState::Asc => {
                if idx == 0 {
                    return self[0] > self[1];
                }
                if idx == self.len() - 1 {
                    return self[idx - 1] > self[idx];
                }
                self[idx - 1] > self[idx] || self[idx] > self[idx + 1]
            }
            SortState::Desc => {
                if idx == 0 {
                    return self[0] < self[1];
                }
                if idx == self.len() - 1 {
                    return self[idx - 1] < self[idx];
                }
                self[idx - 1] < self[idx] || self[idx] < self[idx + 1]
            }
        }
    }
}

impl<T: Ord> EnhVecInner<T> {
    fn sort(&mut self, sorting: &Sorting) {
        // compact() keeps the order, so sort_vec() can trust the current state
        self.compact();
        sort_vec(&mut self.main, &self.state, sorting);
        self.state = match sorting {
            Sorting::Ascending => SortState::Asc,
            Sorting::Descending => SortState::Desc,
            _ => SortState::Unsorted,
        };
    }

    /**
    Insert an element into its sorted position. The data must be known to
    be sorted (state ASC or DESC), and the element is inserted in that order,
    after any equal elements.
    */
    fn insert_sorted(&mut self, element: T) {
        debug_assert!(self.state.is_sorted());
        // whether `a` belongs strictly before `b` in the current order
        let desc: bool = self.state == SortState::Desc;
        let before = |a: &T, b: &T| if desc { a > b } else { a < b };

        // short circuit some common cases
        if self.last().is_none_or(|last: &T| !before(&element, last)) {
            self.main.push(element);
            return;
        }
        if self.first().is_some_and(|x: &T| !before(x, &element)) {
            // head.last() is the first element since head is reversed
            self.head.push(element);
            return;
        }
        let fits_between: bool = match (self.head.first(), self.main.first()) {
            (Some(h), Some(m)) => !before(&element, h) && !before(m, &element),
            _ => false,
        };
        if fits_between {
            // head[0] is the last head element, right before main[0]
            self.head.insert(0, element);
            return;
        }

        // fold head into main and determine the insertion point
        self.compact();
        let idx: usize = match self.main.len() < SEARCH_SIZE_CUTOFF {
            // linear search for "small" vectors
            true => self
                .internal_iter()
                .position(|x: &T| before(&element, x))
                .unwrap_or(self.main.len()),
            // binary search for larger vectors
            false => self.main.partition_point(|x: &T| !before(&element, x)),
        };
        self.main.insert(idx, element);
    }
}

/* --------------------------------- */

impl<T: PartialEq> PartialEq for EnhVecInner<T> {
    fn eq(&self, other: &Self) -> bool {
        // compare the logical order, not how the elements are split between head and main
        self.len() == other.len() && self.internal_iter().eq(other.internal_iter())
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
#[derive(Debug, Clone)]
pub struct EnhVec<T> {
    data: EnhVecInner<T>,
    sort: Sorting,
}

// Implemented by hand, as #[derive(Default)] would require `T: Default`
impl<T> Default for EnhVec<T> {
    fn default() -> Self {
        Self {
            data: EnhVecInner::new(),
            sort: Sorting::None,
        }
    }
}

// Technically PartialEq and PartialOrd bounds are not needed for the
// methods in this block, but we want to restrict the types allowed
// in EnhVec to those that can be compared and sorted.
impl<T: PartialEq + PartialOrd> EnhVec<T> {
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

    /// Insert an element at index. Possibly slow, as it may shift other elements.
    pub fn insert(&mut self, idx: usize, element: T) {
        self.data.insert(idx, element);
    }

    /// Push an element to the end of the [EnhVec].
    pub fn push(&mut self, element: T) {
        self.data.push(element);
    }

    /// Insert an element at the start of the [EnhVec]. Time complexity:
    /// amortized `O(1)`, like `push()`.
    pub fn push_front(&mut self, element: T) {
        self.data.push_front(element);
    }

    /**
    Insert an element at the start of the [EnhVec], without ever moving more
    than a few elements: once the head has `HEAD_SIZE` elements, the new one
    is swapped in and a few others are moved around instead.

    - Advantage: every call is `O(1)`, with no occasional larger moves
      (`push_front()` is `O(1)` only amortized, when a Vec grows).
    - Limitations: the order of the other elements is not preserved, and on
      average it is slower than `push_front()` (2.5-5x in
      `benches/vs_std.rs`). Prefer `push_front()` unless you need the
      worst-case bound and don't care about the order.
    */
    pub fn push_swap_front(&mut self, element: T) {
        self.data.push_swap_front(element);
    }
}

/* --------------------------------- */

// Generic methods for all types
impl<T> EnhVec<T> {
    /**
    Use a comparison Fn to sort the elements. Passthrough to Vec::sort_by().
    Clears the default [Sorting], as the resulting order is custom. Needs no
    [Ord], so also works for floats, e.g. with `sort_by(f64::total_cmp)`.
    */
    pub fn sort_by<F>(&mut self, f: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.data.sort_by(f);
        self.sort = Sorting::None;
    }

    /// Reverse the order of the elements in place. [Sorting] is updated.
    pub fn reverse(&mut self) {
        self.data.reverse();
        match self.data.state {
            SortState::Changed | SortState::Unsorted => {
                self.data.state = SortState::Changed;
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

    /// Clone the elements into a new regular [`Vec<T>`].
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        let mut data: Vec<T> = Vec::with_capacity(self.len());
        data.extend(self.data.head.iter().rev().cloned());
        data.extend_from_slice(&self.data.main);
        data
    }

    /// Consume the [EnhVec] and return the inner [`Vec<T>`].
    /// Time complexity: `O(1)` if there is nothing in the head, else `O(N)`.
    pub fn into_vec(mut self) -> Vec<T> {
        self.data.compact();
        self.data.main
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

    /**
    Remove and return the last element. Time complexity: amortized `O(1)`,
    also mixed with `pop_front()`: when one end runs out, half of the
    elements are moved over from the other end.
    */
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    /// Remove and return the first element. Order of the remaining elements
    /// is preserved. Time complexity: amortized `O(1)`, see `pop()`.
    pub fn pop_front(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    /**
    Remove and return the first element, without ever moving more than one
    element: if the data is not sorted and the head is empty, the last
    element is swapped into the first place (like `Vec::swap_remove(0)`).

    - Advantage: every call is `O(1)` for unsorted data, with no occasional
      larger moves (`pop_front()` is `O(1)` only amortized: when the head
      runs empty, it moves half of the elements over).
    - Limitations: the order of the remaining elements is not preserved, and
      on average it is slower than `pop_front()` (1.3-2x in
      `benches/vs_std.rs`). Sorted data stays sorted, but then this is just
      `pop_front()`. Prefer `pop_front()` unless you need the worst-case
      bound and don't care about the order.
    */
    pub fn swap_pop_front(&mut self) -> Option<T> {
        self.data.swap_pop_front()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    pub fn iter(&self) -> EnhVecIter<'_, T> {
        EnhVecIter::new(&self.data.head, &self.data.main)
    }
    pub fn iter_mut(&mut self) -> EnhVecIterMut<'_, T> {
        self.data.set_changed(); // order of elements could change
        EnhVecIterMut::new(&mut self.data.head, &mut self.data.main)
    }

    /// Move all elements from another [EnhVec] into this one. Maintains the
    /// relative order of the elements. The sort state is set to "None".
    pub fn append(&mut self, other: &mut Self){
        self.data.main.extend(&mut other.data.head.drain(..).rev());
        self.data.main.append(&mut other.data.main);
        self.data.set_changed();
        self.sort = Sorting::None;
    }
}

/* --------------------------------- */

impl<T: PartialOrd> EnhVec<T> {
    /**
    Whether the [EnhVec] is sorted in ascending order. Worst case time
    complexity: `O(N)`, as it may have to compare each element with the next.

    NOTE: an empty or 1-element EnhVec is considered sorted.
    */
    pub fn is_sorted(&self) -> bool {
        self.data.is_sorted()
    }
}

impl<T: Ord> EnhVec<T> {
    /**
    Insert an element into the [EnhVec] in sorted order.

    NOTE: data must be sorted ASC or DESC for this insert to make much sense.
    If the data is not sorted, the insertion point would be more or less
    random, hence in this case we just `push()` the element to the end.
    Data that is sorted both ways (all elements equal) follows the [Sorting]
    given to `sort()` or `new_sorted()`, defaulting to ASC.

    NOTE: if the order is not already known, it is verified first (worst case:
    `O(N)`) and then remembered, so consecutive calls only have to find the
    insertion point (`O(log n)`) and shift the elements after it.

    NOTE: if you need to add many elements, it will likely be faster to push()
    and finally sort() after all the insertions are done, as sorting is approx.
    `O(N log N)`.
    */
    pub fn insert_sorted(&mut self, element: T) {
        if self.data.state.is_unsorted() {
            let state: SortState = match (self.data.is_sorted(), self.data.is_sorted_desc()) {
                (true, true) if self.sort == Sorting::Descending => SortState::Desc,
                (true, _) => SortState::Asc,
                (false, true) => SortState::Desc,
                (false, false) => {
                    self.push(element);
                    return;
                }
            };
            self.data.state = state;
        }
        self.data.insert_sorted(element);
    }

    /**
    Set the default sorting state of the [EnhVec] and sort the data.
    Cheap if the data is already known to be in the requested order: a no-op
    when it is, and a reversal when it is known to be in the opposite order.
    */
    pub fn sort(&mut self, sorting: Sorting) {
        self.data.sort(&sorting);
        self.sort = sorting;
    }

    /// Extend this [EnhVec] from an iterator. If a default [Sorting] is set
    /// (see `sort()` and `new_sorted()`), we will re-sort after the extension.
    pub fn extend_sorted<I>(&mut self, iter: I)
    where I: IntoIterator<Item = T>,
    {
        self.extend(iter);
        if self.sort != Sorting::None {
            self.sort(self.sort);
        }
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

    /**
    Run a closure on each element in ASCending order, for hashing. Unlike
    iterating `as_sorted_asc()`, nothing is collected or sorted if the order
    is known, and otherwise equal elements may come in any order: they are
    equal by [Eq] too (see [Ord]), so they also hash the same.
    */
    fn for_each_asc(&self, f: impl FnMut(&T)) {
        match self.data.state {
            SortState::Asc => self.data.internal_iter().for_each(f),
            SortState::Desc => self.data.internal_iter().rev().for_each(f),
            _ => self.asc_refs().into_iter().for_each(f),
        }
    }

    /// References to the elements in ASCending order, sorted only if the order
    /// is not known. Equal elements come in no particular order.
    fn asc_refs(&self) -> Vec<&T> {
        let mut refs: Vec<&T> = self.as_ref_vec();
        match self.data.state {
            SortState::Asc => {}
            SortState::Desc => refs.reverse(),
            _ => refs.sort_unstable(),
        }
        refs
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
        other.contains_all(self)
    }

    /// Check if the [EnhVec] is a subset of another [EnhVec], ie. all of its
    /// values are in the other one. Order and duplicates are ignored.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.contains_only(other)
    }
    /// Check if the [EnhVec] is a superset of another [EnhVec], ie. all of
    /// the other one's values are in it. Order and duplicates are ignored.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.contains_all(other)
    }
    /// Check if the [EnhVec] is disjoint with another [EnhVec].
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.contains_any(other)
    }
    /// Check if the [EnhVec] has the same set of values as another [EnhVec].
    /// Unlike with `==`, order and duplicates are ignored.
    pub fn is_equal(&self, other: &Self) -> bool {
        self.is_subset(other) && self.is_superset(other)
    }

    /// Check if the [EnhVec] is a proper subset of another [EnhVec], ie.
    /// a subset lacking at least one of the other one's values.
    pub fn is_proper_subset(&self, other: &Self) -> bool {
        self.is_subset(other) && !self.is_superset(other)
    }

    /// Check if the [EnhVec] is a proper superset of another [EnhVec], ie.
    /// a superset with at least one value the other one lacks.
    pub fn is_proper_superset(&self, other: &Self) -> bool {
        self.is_superset(other) && !self.is_subset(other)
    }
    /// Check if the [EnhVec] is a proper subset or superset of another [EnhVec].
    pub fn is_proper(&self, other: &Self) -> bool {
        self.is_proper_subset(other) || self.is_proper_superset(other)
    }
    /**
    Check if the [EnhVec] partially overlaps another [EnhVec]: they share
    at least one value, but both also have values the other one lacks.
    Order and duplicates are ignored.

    Any two non-empty sets are in exactly one of these relations:
    `is_equal()`, `is_proper_subset()`, `is_proper_superset()`,
    `is_disjoint()` or `is_partial_overlap()`.
    */
    pub fn is_partial_overlap(&self, other: &Self) -> bool {
        self.contains_any(other) && !self.is_subset(other) && !self.is_superset(other)
    }
    /// Check if the [EnhVec] is a proper subset and superset of another [EnhVec].
    /// No set can be both, so this checks the closest relation instead.
    #[deprecated(note = "no set is both a proper subset and superset, use is_partial_overlap()")]
    pub fn is_proper_both(&self, other: &Self) -> bool {
        self.is_partial_overlap(other)
    }
    /// Check if the [EnhVec] is a proper subset or superset of another [EnhVec].
    #[deprecated(note = "same as is_proper()")]
    pub fn is_proper_either(&self, other: &Self) -> bool {
        self.is_proper(other)
    }

    /**
    How this and another [EnhVec] relate as sets, see [SetRelation].
    Time complexity: `O(N * M)`, as only [PartialEq] is available. See
    `set_relation_hashed()` and `set_relation_sorted()` for faster ones.
    */
    pub fn set_relation(&self, other: &Self) -> SetRelation {
        let mut rel: SetRelation = SetRelation::default();
        for x in self.data.internal_iter() {
            match other.contains(x) {
                true => rel.shared = true,
                false => rel.left_only = true,
            }
            if rel.shared && rel.left_only {
                break;
            }
        }
        rel.right_only = !self.contains_all(other);
        rel
    }
}

/* --------------------------------- */

/**
How two [EnhVec]s (`self` and `other`, the "left" and the "right" one) relate
as sets of values, ignoring order and duplicates. It is computed in one go by
`set_relation()`, `set_relation_hashed()` or `set_relation_sorted()`, after
which any of the set predicates can be checked for free.

The predicates match the [EnhVec] methods of the same name, e.g.
`a.set_relation_hashed(&b).is_subset() == a.is_subset(&b)`.
*/
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SetRelation {
    /// Some value is in both.
    shared: bool,
    /// Some value is only in the left one.
    left_only: bool,
    /// Some value is only in the right one.
    right_only: bool,
}

impl SetRelation {
    /// All values of the left one are in the right one.
    pub fn is_subset(&self) -> bool {
        !self.left_only
    }
    /// All values of the right one are in the left one.
    pub fn is_superset(&self) -> bool {
        !self.right_only
    }
    /// Same set of values.
    pub fn is_equal(&self) -> bool {
        !self.left_only && !self.right_only
    }
    /// No values in common.
    pub fn is_disjoint(&self) -> bool {
        !self.shared
    }
    /// A subset lacking at least one of the right one's values.
    pub fn is_proper_subset(&self) -> bool {
        !self.left_only && self.right_only
    }
    /// A superset with at least one value the right one lacks.
    pub fn is_proper_superset(&self) -> bool {
        self.left_only && !self.right_only
    }
    /// A proper subset or superset.
    pub fn is_proper(&self) -> bool {
        self.left_only != self.right_only
    }
    /// Some values in common, but both also have values the other one lacks.
    pub fn is_partial_overlap(&self) -> bool {
        self.shared && self.left_only && self.right_only
    }
}

impl<T: Eq + Hash> EnhVec<T> {
    /**
    How this and another [EnhVec] relate as sets, see [SetRelation].
    Time complexity: expected `O(N + M)`, plus building a [HashSet] of each.
    */
    pub fn set_relation_hashed(&self, other: &Self) -> SetRelation {
        let left: HashSet<&T> = self.data.internal_iter().collect();
        let right: HashSet<&T> = other.data.internal_iter().collect();
        let shared: usize = left.iter().filter(|&x| right.contains(x)).count();
        SetRelation {
            shared: shared > 0,
            left_only: shared < left.len(),
            right_only: shared < right.len(),
        }
    }
}

impl<T: Ord> EnhVec<T> {
    /**
    How this and another [EnhVec] relate as sets, see [SetRelation], by
    merging the sorted values. Time complexity: `O(N + M)` if the order of
    both is known (ASC or DESC), else `O(N log N + M log M)`. No hashing.
    */
    pub fn set_relation_sorted(&self, other: &Self) -> SetRelation {
        let (left, right): (Vec<&T>, Vec<&T>) = (self.asc_refs(), other.asc_refs());
        let mut rel: SetRelation = SetRelation::default();
        let (mut i, mut j): (usize, usize) = (0, 0);
        // stop early once all is known
        while i < left.len() && j < right.len() && !rel.is_partial_overlap() {
            match left[i].cmp(right[j]) {
                Ordering::Less => {
                    rel.left_only = true;
                    i += 1;
                }
                Ordering::Greater => {
                    rel.right_only = true;
                    j += 1;
                }
                Ordering::Equal => {
                    rel.shared = true;
                    // skip the duplicates of this value on both sides
                    let value: &T = left[i];
                    while i < left.len() && left[i] == value {
                        i += 1;
                    }
                    while j < right.len() && right[j] == value {
                        j += 1;
                    }
                }
            }
        }
        rel.left_only |= i < left.len();
        rel.right_only |= j < right.len();
        rel
    }
}

/* --------------------------------- */

impl<T: PartialEq> PartialEq for EnhVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq> Eq for EnhVec<T> {}

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

// Extend this EnhVec from an iterator
impl<T> Extend<T> for EnhVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.main.extend(iter);
        self.data.set_changed();
    }
}

// Extend this EnhVec by copying the elements of an iterator of references, like Vec
impl<'a, T: Copy + 'a> Extend<&'a T> for EnhVec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

// Allow `EnhVec::from_iter()` as well as `.collect::<EnhVec<T>>()`
impl<T: PartialEq + PartialOrd> FromIterator<T> for EnhVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new_from(iter.into_iter().collect())
    }
}

/* ############################### Iterators ############################### */

/**
An immutable iterator over the elements of the [EnhVec].

This iterator is a combination of the head and main [Vec]s, with the head
elements returned in the correct order (since the head is basically a reverse
cache in front of the main Vec). The iterator is not sorted, unless the
[EnhVec] is sorted before creating this iterator.
*/
#[derive(Debug)]
pub struct EnhVecIter<'a, T: 'a> {
    head: Rev<Iter<'a, T>>,
    main: Iter<'a, T>,
}

impl<'a, T> EnhVecIter<'a, T> {
    #[rustfmt::skip]
    fn new(head: &'a [T], main: &'a [T]) -> Self {
        Self { head: head.iter().rev(), main: main.iter() }
    }
}

// Implemented by hand, as #[derive(Clone)] would require `T: Clone`
impl<T> Clone for EnhVecIter<'_, T> {
    #[rustfmt::skip]
    fn clone(&self) -> Self {
        Self { head: self.head.clone(), main: self.main.clone() }
    }
}

/* --------------------------------- */

/// A mutable iterator over the elements of the [EnhVec].
#[derive(Debug)]
pub struct EnhVecIterMut<'a, T: 'a> {
    head: Rev<IterMut<'a, T>>,
    main: IterMut<'a, T>,
}

impl<'a, T> EnhVecIterMut<'a, T> {
    #[rustfmt::skip]
    fn new(head: &'a mut [T], main: &'a mut [T]) -> Self {
        Self { head: head.iter_mut().rev(), main: main.iter_mut() }
    }
}

/* --------------------------------- */

/// Common iterator trait impls for [EnhVecIter] and [EnhVecIterMut].
macro_rules! impl_enhvec_iter {
    ($iter:ident, $item:ty) => {
        impl<'a, T> Iterator for $iter<'a, T> {
            type Item = $item;

            fn next(&mut self) -> Option<Self::Item> {
                self.head.next().or_else(|| self.main.next())
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len: usize = self.head.len() + self.main.len();
                (len, Some(len))
            }

            // separate loops over head and main (like Chain), used by e.g. for_each()
            fn fold<B, F>(self, init: B, mut f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                let acc: B = self.head.fold(init, &mut f);
                self.main.fold(acc, f)
            }
        }

        impl<'a, T> DoubleEndedIterator for $iter<'a, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.main.next_back().or_else(|| self.head.next_back())
            }

            fn rfold<B, F>(self, init: B, mut f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                let acc: B = self.main.rfold(init, &mut f);
                self.head.rfold(acc, f)
            }
        }

        impl<T> ExactSizeIterator for $iter<'_, T> {}
        impl<T> FusedIterator for $iter<'_, T> {}
    };
}

impl_enhvec_iter!(EnhVecIter, &'a T);
impl_enhvec_iter!(EnhVecIterMut, &'a mut T);

/* --------------------------------- */

impl<T> IntoIterator for EnhVec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a EnhVec<T> {
    type Item = &'a T;
    type IntoIter = EnhVecIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut EnhVec<T> {
    type Item = &'a mut T;
    type IntoIter = EnhVecIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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

    Like the standard slice hash, the length is hashed first, so that e.g.
    the tuples `([1, 2], [3])` and `([1], [2, 3])` do not collide.
    */
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        self.for_each_asc(|elem: &T| elem.hash(state));
    }
}

impl<T: Ord + Xxh3Hashable> Xxh3Hashable for EnhVec<T> {
    /**
    This method is used to hash the elements in a stable, repeatable way.
    Internally it works just like the standard `hash()` method, ie. it
    updates the state of the given hasher with each element in turn.

    The element in question must implement the [Xxh3Hashable] trait and
    actually hash itself using the `xxh3()` method of course.

    The length is hashed first (see [Hash]), as a little-endian `u64` to
    keep the result the same across platforms.
    */
    #[inline]
    fn xxh3<H: Hasher>(&self, state: &mut H) {
        state.write(&(self.len() as u64).to_le_bytes());
        self.for_each_asc(|elem: &T| elem.xxh3(state));
    }

    /**
    This method is used to hash the elements in a stable, repeatable way.
    In contrast to `xxh3()`, this method returns the final u64 hash value.
    */
    #[inline]
    fn xxh3_digest(&self) -> u64 {
        let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
        self.xxh3(&mut hasher);
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

/* --------------------------------- */

impl<T: Copy + Eq + Hash> EnhVec<T> {
    /// Return the mode (most common) value of the elements. If several values
    /// are equally common, the one that appears first is returned.
    pub fn mode(&self) -> Option<T> {
        let mut counts: HashMap<T, usize> = HashMap::new();
        for &item in self.data.internal_iter() {
            *counts.entry(item).or_insert(0) += 1;
        }

        // pick by the element order, so the result does not depend on the HashMap's
        let max: usize = counts.values().copied().max()?;
        self.data
            .internal_iter()
            .copied()
            .find(|item: &T| counts[item] == max)
    }

    /// Return the distinct (unique) elements in the order of their first
    /// appearance, or sorted if requested.
    pub fn distinct(&self, sorted: Option<Sorting>) -> EnhVec<T>
    where
        T: Copy + Eq + Hash + Ord,
    {
        let mut seen: HashSet<T> = HashSet::new();
        let unique: Vec<T> = self
            .data
            .internal_iter()
            .copied()
            .filter(|&x| seen.insert(x))
            .collect();
        let mut result: EnhVec<T> = EnhVec::new_from(unique);
        if self.data.state.is_sorted() {
            // dropping duplicates keeps a known order
            result.data.state = self.data.state.clone();
        }
        if let Some(sorting) = sorted {
            result.sort(sorting);
        }
        result
    }
}

/* --------------------------------- */

impl<T: Integer> EnhVec<T> {
    /**
    Return the range (max - min) of the elements. `None` if there are none,
    or if the range does not fit in `T` (e.g. `i8` values -128 and 127).
    Time complexity: `O(1)` if the order is known, else `O(N)`.
    */
    pub fn range(&self) -> Option<T> {
        let (first, last): (T, T) = (*self.first()?, *self.last()?);
        let (min, max): (T, T) = match self.data.state {
            SortState::Asc => (first, last),
            SortState::Desc => (last, first),
            // min and max in a single pass
            _ => self
                .data
                .internal_iter()
                .fold((first, first), |(min, max), &x| (min.min(x), max.max(x))),
        };
        max.checked_sub(min)
    }

    /**
    The element at position `idx` of the data in ASCending order, and if
    `with_next`, the one after it. Time complexity: `O(1)` if the order is
    known, else `O(N)` (selection on a copy of the data, not a full sort).
    */
    fn select_asc(&self, idx: usize, with_next: bool) -> (T, Option<T>) {
        if !self.data.state.is_sorted() {
            return select_by(&mut self.to_vec(), idx, with_next, T::cmp);
        }
        let len: usize = self.len();
        let at_asc = |i: usize| match self.data.state {
            SortState::Desc => self.data[len - 1 - i],
            _ => self.data[i],
        };
        let next: Option<T> = (with_next && idx + 1 < len).then(|| at_asc(idx + 1));
        (at_asc(idx), next)
    }

    /// Return the median (aka. the middle) value of the elements.
    /// Time complexity: `O(1)` if the order is known, else `O(N)`.
    pub fn median(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mid: usize = self.len() / 2;

        if self.len().is_multiple_of(2) {
            // unlike `(a + b) / 2`, midpoint() cannot overflow
            let (below, above): (T, Option<T>) = self.select_asc(mid - 1, true);
            above.map(|above: T| T::midpoint(below, above))
        } else {
            Some(self.select_asc(mid, false).0)
        }
    }

    /// Return the average (mean) value of the elements.
    /// The elements are summed as `i128`, so the sum cannot overflow `T`.
    pub fn average(&self) -> Option<f64>
    where
        T: Into<i128>,
    {
        if self.is_empty() {
            return None;
        }

        let sum: i128 = self.data.internal_iter().map(|&x: &T| x.into()).sum();
        Some(sum as f64 / self.len() as f64)
    }

    /// Return the product of all elements. For empty EnhVec, `product == 1`.
    /// Multiplications are performed as `i128`, and `None` means it overflowed.
    pub fn product(&self) -> Option<i128>
    where
        T: Into<i128>,
    {
        self.data
            .internal_iter()
            .try_fold(1, |acc: i128, &x| acc.checked_mul(x.into()))
    }

    /// Return the population variance of the elements, ie. the mean of
    /// the squared deviations from the mean (divided by `N`).
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

    /// Return the percentile value of the elements. NOTE: `0.0 <= p <= 1.0`
    pub fn percentile(&self, p: f64) -> Option<T> {
        // written as !contains() so that a NaN `p` is rejected as well
        if self.is_empty() || !(0.0..=1.0).contains(&p) {
            return None;
        }

        let index: usize = (p * (self.len() - 1) as f64).round() as usize;
        Some(self.select_asc(index, false).0)
    }
}

/* --------------------------------- */

impl<T: Float> EnhVec<T> {
    /// Return the median (aka. the middle) value of the elements.
    /// Floating point compatible version. NaNs are ordered by `total_cmp()`.
    pub fn median_fp(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let mut data: Vec<T> = self.to_vec();
        let mid: usize = data.len() / 2;

        if data.len().is_multiple_of(2) {
            // unlike `(a + b) / 2`, midpoint() cannot overflow to infinity
            let (below, above): (T, Option<T>) = select_by(&mut data, mid - 1, true, T::total_cmp);
            above.map(|above: T| T::midpoint(below, above))
        } else {
            Some(select_by(&mut data, mid, false, T::total_cmp).0)
        }
    }

    /// Return the average (mean) value of the elements.
    /// Floating point version.
    pub fn average_fp(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let len: T = T::from_usize(self.len()).unwrap();
        let sum: T = self.data.internal_iter().copied().sum();
        if sum.is_finite() {
            return Some(sum / len);
        }
        // the sum overflowed (or there are infinities/NaNs): sum pre-divided values
        Some(self.data.internal_iter().map(|&x: &T| x / len).sum())
    }

    /// Return the product of all elements. Floating point version.
    pub fn product_fp(&self) -> T {
        self.data.internal_iter().fold(T::one(), |acc, &x| acc * x)
    }

    /// Return the population variance of the elements (divided by `N`).
    /// Floating point version.
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
            / T::from_usize(self.len()).unwrap();
        Some(variance)
    }

    /// Return the standard deviation of the elements. Floating point version.
    pub fn stdev_fp(&self) -> Option<T> {
        self.variance_fp().map(|v: T| v.sqrt())
    }

    /// Return the percentile value of the elements. Floating point version.
    /// NOTE: `0.0 <= p <= 1.0`. NaNs are ordered by `total_cmp()`.
    pub fn percentile_fp(&self, p: f64) -> Option<T> {
        // written as !contains() so that a NaN `p` is rejected as well
        if self.is_empty() || !(0.0..=1.0).contains(&p) {
            return None;
        }

        let index: usize = (p * (self.len() - 1) as f64).round() as usize;
        Some(select_by(&mut self.to_vec(), index, false, T::total_cmp).0)
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
    /// `(self + rhs) / 2` without overflow, rounded towards zero.
    fn midpoint(self, rhs: Self) -> Self;
    /// `self - rhs`, or `None` on overflow.
    fn checked_sub(self, rhs: Self) -> Option<Self>;
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
                #[inline]
                fn midpoint(self, rhs: Self) -> Self { self.midpoint(rhs) }
                #[inline]
                fn checked_sub(self, rhs: Self) -> Option<Self> { self.checked_sub(rhs) }
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
                #[inline]
                fn midpoint(self, rhs: Self) -> Self { self.midpoint(rhs) }
                #[inline]
                fn checked_sub(self, rhs: Self) -> Option<Self> { self.checked_sub(rhs) }
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
    /// `(self + rhs) / 2` without overflowing to infinity.
    fn midpoint(self, rhs: Self) -> Self;
    /// Total ordering, including NaN (see [f64::total_cmp]).
    fn total_cmp(&self, other: &Self) -> Ordering;
    /// Neither infinite nor NaN.
    fn is_finite(self) -> bool;
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
                #[inline]
                fn midpoint(self, rhs: Self) -> Self { self.midpoint(rhs) }
                #[inline]
                fn total_cmp(&self, other: &Self) -> Ordering { self.total_cmp(other) }
                #[inline]
                fn is_finite(self) -> bool { self.is_finite() }
            }
        )*
    }
}

impl_float!(f32, f64);

/* ########################### Utility functions ########################### */

/**
The element a full sort of `v` by `cmp` would put at `idx`, and if `with_next`,
the one it would put right after it. Reorders `v`. Time complexity: `O(N)`.
*/
fn select_by<T: Copy, F>(v: &mut [T], idx: usize, with_next: bool, mut cmp: F) -> (T, Option<T>)
where
    F: FnMut(&T, &T) -> Ordering,
{
    let (_, nth, above) = v.select_nth_unstable_by(idx, &mut cmp);
    let next: Option<T> = match with_next {
        true => above.iter().copied().min_by(|a: &T, b: &T| cmp(a, b)),
        false => None,
    };
    (*nth, next)
}

/// Sort a vector in place, based on the current and desired sorting state.
fn sort_vec<T: Ord>(v: &mut [T], state: &SortState, desired: &Sorting) {
    // short circuit no-ops
    let noop: bool = matches!(
        (state, desired),
        (_, Sorting::None)
            | (SortState::Asc, Sorting::Ascending)
            | (SortState::Desc, Sorting::Descending)
    );
    if noop || v.len() < 2 {
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
    use std::{collections::VecDeque, hash::DefaultHasher, iter::from_fn};

    const PI_LEN: usize = 16;
    const PI_SUM: u32 = 80;
    const PI_ARR: [u32; PI_LEN] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
    const PI_ASC: [u32; PI_LEN] = [1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9];
    const PI_DESC: [u32; PI_LEN] = [9, 9, 9, 8, 7, 6, 5, 5, 5, 4, 3, 3, 3, 2, 1, 1];
    const FP_ARR: [f64; 7] = [-999.0, 1.0, 2.0, 3.0, 4.0, 5.0, 999.0];
    const XTRA: u32 = 99;
    const EPSILON: f64 = 1e-10;

    /// Minimal [Xxh3Hashable] element for the hashing tests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct XxhU32(u32);

    impl Xxh3Hashable for XxhU32 {
        fn xxh3<H: Hasher>(&self, state: &mut H) {
            state.write(&self.0.to_le_bytes());
        }
        fn xxh3_digest(&self) -> u64 {
            let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
            self.xxh3(&mut hasher);
            hasher.finish()
        }
    }

    fn std_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher: DefaultHasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn xxh3_vec(values: &[u32]) -> EnhVec<XxhU32> {
        EnhVec::from_iter(values.iter().map(|&x: &u32| XxhU32(x)))
    }

    /// Check the order based statistics of `ev` against a full sort of `data`.
    fn check_order_stats(ev: &EnhVec<u32>, data: &[u32], msg: &str) {
        let mut sorted: Vec<u32> = data.to_vec();
        sorted.sort();
        let (n, mid): (usize, usize) = (sorted.len(), sorted.len() / 2);
        let median: u32 = match n % 2 {
            0 => sorted[mid - 1].midpoint(sorted[mid]),
            _ => sorted[mid],
        };
        assert_eq!(ev.median(), Some(median), "median, {msg}");
        assert_eq!(ev.range(), Some(sorted[n - 1] - sorted[0]), "range, {msg}");
        for p in (0..=20).map(|i: u32| i as f64 / 20.0) {
            let idx: usize = (p * (n - 1) as f64).round() as usize;
            assert_eq!(ev.percentile(p), Some(sorted[idx]), "percentile {p}, {msg}");
        }
        let hash: u64 = std_hash(&EnhVec::from_iter(sorted));
        assert_eq!(std_hash(ev), hash, "Hash, {msg}");
    }

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
        test.insert(0, XTRA);
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
        ev.for_each_if(|x: &u32| sum_if += x, |x: &u32| x.is_multiple_of(2));
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
        assert_eq!(ev.product(), Some(prod));
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
        let diff: f32 = ev.average_fp().unwrap() - 15.0 / 7.0;
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
        assert_eq!(ev.product(), Some(1), "product is not one");
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
        assert_eq!(ev.product(), Some(purpose.into()), "product");
        assert_eq!(ev.range(), Some(0), "range");
        assert_eq!(ev.median(), Some(purpose), "median");
        assert_eq!(ev.mode(), Some(purpose), "mode");
        assert_eq!(ev.average(), Some(purpose as f64), "average");
        assert_eq!(ev.percentile(0.01), Some(purpose), "percentile 0.01");
        assert_eq!(ev.percentile(0.50), Some(purpose), "percentile 0.50");
        assert_eq!(ev.percentile(0.99), Some(purpose), "percentile 0.99");
        assert!(ev.is_sorted(), "is_sorted");
    }

    #[test]
    fn test_push_front_compact_desc() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([3, 2, 1]);
        ev.sort(Sorting::Descending);
        // every push keeps the DESC order
        let top: u32 = 4 + HEAD_SIZE as u32;
        (4..=top).for_each(|x: u32| ev.push_front(x));

        let test: Vec<u32> = (1..=top).rev().collect();
        assert_eq!(ev.to_vec(), test);
        assert_eq!(ev.as_sorted_desc(), test.iter().collect::<Vec<&u32>>());
    }

    #[test]
    #[rustfmt::skip]
    fn test_insert_into_head() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([10, 11]);
        ev.push_front(2);
        ev.push_front(1);
        ev.insert(0, 0);
        assert_eq!(ev.to_vec(), vec![0, 1, 2, 10, 11], "insert at front");
        ev.insert(2, XTRA);
        assert_eq!(ev.to_vec(), vec![0, 1, XTRA, 2, 10, 11], "insert inside head");
        ev.insert(4, XTRA);
        assert_eq!(ev.to_vec(), vec![0, 1, XTRA, 2, XTRA, 10, 11], "insert at head/main junction");
    }

    #[test]
    fn test_reverse_with_head() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([3, 4]);
        ev.push_front(2);
        ev.push_front(1);
        ev.reverse();
        assert_eq!(ev.to_vec(), vec![4, 3, 2, 1]);

        let mut sorted: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        sorted.sort(Sorting::Ascending);
        sorted.push_front(0); // keeps ASC order, lands in the head
        sorted.reverse();
        let mut test: Vec<u32> = Vec::from_iter(PI_DESC);
        test.push(0);
        assert_eq!(sorted.to_vec(), test);
        assert_eq!(sorted.as_sorted_desc(), test.iter().collect::<Vec<&u32>>());
    }

    #[test]
    fn test_pop_front_keeps_order() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.push_front(XTRA);
        let popped: Vec<u32> = from_fn(|| ev.pop_front()).collect();
        let mut test: Vec<u32> = Vec::from_iter(PI_ARR);
        test.insert(0, XTRA);
        assert_eq!(popped, test, "unsorted");

        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.sort_by(|a: &u32, b: &u32| b.cmp(a));
        let popped: Vec<u32> = from_fn(|| ev.pop_front()).collect();
        assert_eq!(popped, Vec::from_iter(PI_DESC), "custom sort_by() order");
    }

    #[test]
    fn test_swap_pop_front() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([1, 2, 3, 4]);
        assert_eq!(ev.swap_pop_front(), Some(1));
        assert_eq!(ev.to_vec(), vec![4, 2, 3], "unsorted: last one swapped in");

        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.sort(Sorting::Ascending);
        let popped: Vec<u32> = from_fn(|| ev.swap_pop_front()).collect();
        assert_eq!(popped, Vec::from_iter(PI_ASC), "sorted: order kept");
    }

    #[test]
    fn test_insert_sorted_head() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([7, 9]);
        ev.insert_sorted(5); // new first element
        ev.insert_sorted(6); // between head and main
        ev.insert_sorted(3); // new first element
        ev.insert_sorted(4); // < main[0], but also < the last head element
        assert_eq!(ev.to_vec(), vec![3, 4, 5, 6, 7, 9]);

        // all elements in the head, main is empty
        let mut ev: EnhVec<u32> = EnhVec::new();
        ev.push_front(5);
        ev.insert_sorted(3);
        ev.insert_sorted(4);
        ev.insert_sorted(XTRA);
        assert_eq!(ev.to_vec(), vec![3, 4, 5, XTRA]);
    }

    #[test]
    fn test_insert_sorted_many() {
        // deterministic pseudo-random values with duplicates, enough for binary search
        let values: Vec<u32> = (0..200).map(|i: u32| (i * 7919) % 101).collect();
        let mut asc: EnhVec<u32> = EnhVec::new();
        let mut desc: EnhVec<u32> = EnhVec::new_sorted(Sorting::Descending);
        values.iter().for_each(|&x: &u32| {
            asc.insert_sorted(x);
            desc.insert_sorted(x);
        });

        let mut test: Vec<u32> = values.clone();
        test.sort();
        assert_eq!(asc.to_vec(), test, "ASC");
        test.reverse();
        assert_eq!(desc.to_vec(), test, "DESC");
    }

    #[test]
    #[rustfmt::skip]
    fn test_insert_sorted_desc() {
        let mut ev: EnhVec<i32> = EnhVec::from_iter([5]);
        ev.sort(Sorting::Descending);
        ev.insert_sorted(7);
        assert_eq!(ev.to_vec(), vec![7, 5], "DESC state, 1 element");
        assert_eq!(ev.percentile(0.0), Some(5), "DESC state, 1 element: min");

        let mut ev: EnhVec<i32> = EnhVec::from_iter([2, 2, 2]);
        ev.sort(Sorting::Descending);
        ev.insert_sorted(5);
        ev.insert_sorted(1);
        assert_eq!(ev.to_vec(), vec![5, 2, 2, 2, 1], "DESC state, all equal");
        assert_eq!(ev.as_sorted_desc(), vec![&5, &2, &2, &2, &1], "DESC state, all equal");

        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_DESC);
        ev.insert_sorted(4);
        let mut test: Vec<u32> = Vec::from_iter(PI_DESC);
        test.insert(10, 4);
        assert_eq!(ev.to_vec(), test, "DESC data, unknown state");
    }

    #[test]
    fn test_sort_after_mutation() {
        let assert_asc = |ev: &EnhVec<u32>, msg: &str| {
            let mut test: Vec<u32> = ev.to_vec();
            test.sort();
            assert_eq!(ev.to_vec(), test, "{msg}");
        };
        let mut ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        ev.sort(Sorting::Ascending);
        ev.push(0);
        ev.sort(Sorting::Ascending);
        assert_asc(&ev, "after push()");
        ev[0] = XTRA;
        ev.sort(Sorting::Ascending);
        assert_asc(&ev, "after IndexMut");
        ev.sort_by(|a: &u32, b: &u32| b.cmp(a));
        ev.sort(Sorting::Ascending);
        assert_asc(&ev, "after sort_by()");
        ev.iter_mut().for_each(|x: &mut u32| *x = XTRA - *x);
        ev.sort(Sorting::Ascending);
        assert_asc(&ev, "after iter_mut()");
    }

    #[test]
    #[rustfmt::skip]
    fn test_iter_mut_resets_state() {
        let mut ev: EnhVec<i32> = EnhVec::from_iter([1, 2, 3]);
        ev.sort(Sorting::Ascending);
        ev.iter_mut().for_each(|x: &mut i32| *x = -*x);
        assert!(!ev.is_sorted(), "is_sorted after iter_mut()");
        assert_eq!(ev.as_sorted_asc(), vec![&-3, &-2, &-1], "as_sorted_asc after iter_mut()");
        assert_eq!(ev.percentile(0.0), Some(-3), "min after iter_mut()");

        ev.sort(Sorting::Ascending);
        for x in &mut ev {
            *x = -*x;
        }
        assert!(!ev.is_sorted(), "is_sorted after &mut iteration");
        assert_eq!(ev.as_sorted_asc(), vec![&1, &2, &3], "as_sorted_asc after &mut iteration");
    }

    #[test]
    fn test_extend_sorted() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([1, 5]);
        ev.sort(Sorting::Ascending);
        ev.extend_sorted([3, 0]);
        assert_eq!(ev.to_vec(), vec![0, 1, 3, 5], "ASC");
        ev.sort(Sorting::Descending);
        ev.extend_sorted([4, XTRA]);
        assert_eq!(ev.to_vec(), vec![XTRA, 5, 4, 3, 1, 0], "DESC");

        let mut ev: EnhVec<u32> = EnhVec::new_sorted(Sorting::Ascending);
        ev.extend_sorted(PI_ARR);
        assert_eq!(ev.to_vec(), Vec::from_iter(PI_ASC), "new_sorted()");
        ev.extend(PI_ARR);
        ev.sort(Sorting::Ascending);
        assert_eq!(ev.len(), 2 * PI_LEN, "sort() after extend()");
        assert!(ev.is_sorted(), "sort() after extend()");
    }

    #[test]
    #[rustfmt::skip]
    fn test_set_operations() {
        let small: EnhVec<u32> = EnhVec::from_iter([1, 2]);
        let big: EnhVec<u32> = EnhVec::from_iter([3, 2, 1]);
        let dups: EnhVec<u32> = EnhVec::from_iter([2, 1, 2, 1, 1]);
        let other: EnhVec<u32> = EnhVec::from_iter([2, XTRA]);
        let ones: EnhVec<u32> = EnhVec::from_iter([1, 1, 1]);

        assert!(big.contains_all(&small) && !small.contains_all(&big), "contains_all");
        assert!(ones.contains_only(&big) && !big.contains_only(&ones), "contains_only");
        assert!(small.is_subset(&big) && !big.is_subset(&small), "is_subset");
        assert!(big.is_superset(&small) && !small.is_superset(&big), "is_superset");
        assert!(!small.is_superset(&other) && !small.is_subset(&other), "partial overlap");
        assert!(!small.is_disjoint(&other) && ones.is_disjoint(&other), "is_disjoint");
        assert!(small.is_equal(&dups) && dups.is_equal(&small), "is_equal ignores duplicates");
        assert!(!big.is_equal(&ones) && !ones.is_equal(&big), "is_equal");
        assert!(small.is_proper_subset(&big) && !small.is_proper_subset(&dups), "is_proper_subset");
        assert!(big.is_proper_superset(&small) && !dups.is_proper_superset(&small), "is_proper_superset");
        assert!(small.is_proper(&big) && big.is_proper(&small), "is_proper");
        assert!(!small.is_proper(&other) && !small.is_proper(&dups), "is_proper");
        assert!(small.is_partial_overlap(&other) && other.is_partial_overlap(&small), "is_partial_overlap");
        assert!(!small.is_partial_overlap(&big) && !small.is_partial_overlap(&dups), "is_partial_overlap, comparable");
        assert!(!ones.is_partial_overlap(&other), "is_partial_overlap, disjoint");

        // any two non-empty sets are in exactly one of these relations
        let sets: [&EnhVec<u32>; 5] = [&small, &big, &dups, &other, &ones];
        for (a, b) in sets.iter().flat_map(|&a| sets.iter().map(move |&b| (a, b))) {
            let relations: [bool; 5] = [
                a.is_equal(b), a.is_proper_subset(b), a.is_proper_superset(b),
                a.is_disjoint(b), a.is_partial_overlap(b),
            ];
            let found: usize = relations.iter().filter(|&&r| r).count();
            assert_eq!(found, 1, "{:?} vs {:?}: {relations:?}", a.to_vec(), b.to_vec());
        }
    }

    #[test]
    #[rustfmt::skip]
    #[allow(deprecated)]
    fn test_set_operation_aliases() {
        let small: EnhVec<u32> = EnhVec::from_iter([1, 2]);
        let big: EnhVec<u32> = EnhVec::from_iter([3, 2, 1]);
        let other: EnhVec<u32> = EnhVec::from_iter([2, XTRA]);
        assert!(small.is_proper_both(&other) && !small.is_proper_both(&big), "is_proper_both");
        assert!(small.is_proper_either(&big) && !small.is_proper_either(&other), "is_proper_either");
    }

    #[test]
    fn test_ops_against_vecdeque() {
        // deterministic pseudo-random operations (xorshift), checked against a VecDeque
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: usize| -> usize {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % bound as u64) as usize
        };
        let mut ev: EnhVec<usize> = EnhVec::new();
        let mut model: VecDeque<usize> = VecDeque::new();

        for step in 0..5000 {
            let x: usize = next(1000);
            match next(12) {
                0..=2 => {
                    ev.push(x);
                    model.push_back(x);
                }
                3..=5 => {
                    ev.push_front(x);
                    model.push_front(x);
                }
                6 => assert_eq!(ev.pop(), model.pop_back(), "pop, step {step}"),
                7 => assert_eq!(ev.pop_front(), model.pop_front(), "pop_front, step {step}"),
                8 => {
                    let idx: usize = next(model.len() + 1);
                    ev.insert(idx, x);
                    model.insert(idx, x);
                }
                9 => {
                    ev.reverse();
                    model.make_contiguous().reverse();
                }
                10 => {
                    ev.sort(Sorting::Ascending);
                    model.make_contiguous().sort();
                }
                _ => {
                    ev.sort(Sorting::Descending);
                    model.make_contiguous().sort_by(|a, b| b.cmp(a));
                }
            }
            assert!(ev.iter().eq(model.iter()), "elements, step {step}");
            assert_eq!(ev.len(), model.len(), "len, step {step}");
            let sorted: bool = model.iter().is_sorted();
            assert_eq!(ev.is_sorted(), sorted, "is_sorted, step {step}");
            let sorted_desc: bool = model.iter().rev().is_sorted();
            assert_eq!(ev.data.is_sorted_desc(), sorted_desc, "DESC, step {step}");
            if !model.is_empty() {
                let idx: usize = next(model.len());
                assert_eq!(ev[idx], model[idx], "index {idx}, step {step}");
            }
        }
    }

    #[test]
    fn test_order_stats_known_order() {
        // even and odd length
        for data in [&PI_ARR[..], &PI_ARR[1..]] {
            let unsorted: EnhVec<u32> = EnhVec::from_iter(data.iter().copied());
            check_order_stats(&unsorted, data, "unknown order");

            let mut asc: EnhVec<u32> = unsorted.clone();
            asc.sort(Sorting::Ascending);
            check_order_stats(&asc, data, "ASC");
            asc.push_front(0); // keeps ASC, lands in the head
            assert!(asc.data.state == SortState::Asc && !asc.data.head.is_empty());
            check_order_stats(&asc, &[data, &[0]].concat(), "ASC with head");

            let mut desc: EnhVec<u32> = unsorted.clone();
            desc.sort(Sorting::Descending);
            check_order_stats(&desc, data, "DESC");
            desc.push_front(XTRA); // keeps DESC, lands in the head
            assert!(desc.data.state == SortState::Desc && !desc.data.head.is_empty());
            check_order_stats(&desc, &[data, &[XTRA]].concat(), "DESC with head");
        }
    }

    #[test]
    fn test_default_without_default_elements() {
        // Ordering has no Default impl, which #[derive(Default)] used to require
        #[derive(Default)]
        struct Holder {
            ev: EnhVec<Ordering>,
        }
        let holder: Holder = Holder::default();
        assert!(holder.ev.is_empty());
    }

    #[test]
    fn test_sort_by_floats() {
        let mut ev: EnhVec<f64> = EnhVec::from_iter([3.0, f64::NAN, -1.0, 2.0]);
        ev.sort_by(f64::total_cmp);
        let sorted: Vec<f64> = ev.to_vec();
        assert_eq!(sorted[..3], [-1.0, 2.0, 3.0]);
        assert!(sorted[3].is_nan());
    }

    #[test]
    #[rustfmt::skip]
    fn test_mode_and_distinct_deterministic() {
        // equally common values: the first one to appear wins
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.mode(), Some(3), "mode");
        let mut tie: EnhVec<u32> = EnhVec::from_iter([1, 2, 1, 2]);
        assert_eq!(tie.mode(), Some(1), "mode");
        tie.reverse();
        assert_eq!(tie.mode(), Some(2), "mode, reversed");

        let distinct: EnhVec<u32> = ev.distinct(None);
        assert_eq!(distinct.to_vec(), vec![3, 1, 4, 5, 9, 2, 6, 8, 7], "first appearance order");
        let mut asc: EnhVec<u32> = ev.clone();
        asc.sort(Sorting::Ascending);
        let distinct: EnhVec<u32> = asc.distinct(None);
        assert_eq!(distinct.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9], "sorted source");
        assert_eq!(distinct.data.state, SortState::Asc, "known order kept");
    }

    #[test]
    fn test_range_overflow() {
        let ev: EnhVec<i8> = EnhVec::from_iter([-128, 127]);
        assert_eq!(ev.range(), None, "range does not fit in i8");
        let ev: EnhVec<i8> = EnhVec::from_iter([-100, 27, 5]);
        assert_eq!(ev.range(), Some(127), "range fits in i8");
        let mut ev: EnhVec<u8> = EnhVec::from_iter([0, 255, 7]);
        assert_eq!(ev.range(), Some(255), "unknown order");
        ev.sort(Sorting::Descending);
        assert_eq!(ev.range(), Some(255), "DESC");
    }

    #[test]
    fn test_average_fp_overflow() {
        let ev: EnhVec<f64> = EnhVec::from_iter([f64::MAX, f64::MAX]);
        assert_eq!(ev.average_fp(), Some(f64::MAX), "sum overflows");
        let ev: EnhVec<f64> = EnhVec::from_iter([1e308, 1e308, 1e308]);
        let diff: f64 = ev.average_fp().unwrap() / 1e308 - 1.0;
        assert!(diff.abs() < EPSILON, "sum overflows, relative diff {diff}");
        let ev: EnhVec<f64> = EnhVec::from_iter([1.0, f64::INFINITY]);
        assert_eq!(ev.average_fp(), Some(f64::INFINITY), "infinite element");
        let ev: EnhVec<f64> = EnhVec::from_iter([1.0, f64::NAN]);
        assert!(ev.average_fp().is_some_and(f64::is_nan), "NaN element");
    }

    #[test]
    #[rustfmt::skip]
    fn test_iterators() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([3, 4, 5]);
        ev.push_front(2);
        ev.push_front(1); // head [2, 1], main [3, 4, 5]
        let test: Vec<u32> = vec![1, 2, 3, 4, 5];

        let rev: Vec<u32> = ev.iter().rev().copied().collect();
        assert_eq!(rev, test.iter().rev().copied().collect::<Vec<u32>>(), "rev()");
        let mut iter: EnhVecIter<'_, u32> = ev.iter();
        let ends: [Option<&u32>; 4] = [iter.next(), iter.next_back(), iter.next(), iter.next_back()];
        assert_eq!(ends, [Some(&1), Some(&5), Some(&2), Some(&4)], "both ends");
        let rest: EnhVecIter<'_, u32> = iter.clone();
        assert_eq!(rest.copied().collect::<Vec<u32>>(), vec![3], "clone()");
        assert_eq!((iter.next(), iter.next(), iter.next_back()), (Some(&3), None, None), "fused");

        assert_eq!(ev.iter().fold(0, |acc: u32, x: &u32| acc * 10 + x), 12345, "fold()");
        assert_eq!(ev.iter().rfold(0, |acc: u32, x: &u32| acc * 10 + x), 54321, "rfold()");
        ev.iter_mut().rev().enumerate().for_each(|(i, x): (usize, &mut u32)| *x += i as u32);
        assert_eq!(ev.to_vec(), vec![5, 5, 5, 5, 5], "iter_mut().rev()");
    }

    #[test]
    fn test_extend() {
        let mut ev: EnhVec<u32> = EnhVec::from_iter([1, 2]);
        ev.extend([3, 4]);
        ev.extend(&[5, 6]);
        ev.extend([7].iter());
        assert_eq!(ev.to_vec(), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    #[rustfmt::skip]
    fn test_set_relation() {
        let data: [&[u32]; 9] = [
            &[], &[1], &[1, 1, 2], &[2, 1], &[3, 2, 1], &[2, XTRA], &[1, 1, 1], &[4, 5], &[5, 5, 4],
        ];
        // every combination, with unknown, ASC and DESC order
        let mut vecs: Vec<EnhVec<u32>> = Vec::new();
        for values in data {
            let ev: EnhVec<u32> = EnhVec::from_iter(values.iter().copied());
            let (mut asc, mut desc): (EnhVec<u32>, EnhVec<u32>) = (ev.clone(), ev.clone());
            asc.sort(Sorting::Ascending);
            desc.sort(Sorting::Descending);
            vecs.extend([ev, asc, desc]);
        }

        for (a, b) in vecs.iter().flat_map(|a| vecs.iter().map(move |b| (a, b))) {
            let msg: String = format!("{:?} vs {:?}", a.to_vec(), b.to_vec());
            let rel: SetRelation = a.set_relation(b);
            assert_eq!(a.set_relation_hashed(b), rel, "hashed, {msg}");
            assert_eq!(a.set_relation_sorted(b), rel, "sorted, {msg}");

            let expected: [bool; 8] = [
                a.is_subset(b), a.is_superset(b), a.is_equal(b), a.is_disjoint(b),
                a.is_proper_subset(b), a.is_proper_superset(b), a.is_proper(b),
                a.is_partial_overlap(b),
            ];
            let found: [bool; 8] = [
                rel.is_subset(), rel.is_superset(), rel.is_equal(), rel.is_disjoint(),
                rel.is_proper_subset(), rel.is_proper_superset(), rel.is_proper(),
                rel.is_partial_overlap(),
            ];
            assert_eq!(found, expected, "predicates, {msg}");
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_compact_capacity() {
        let mut ev: EnhVec<u32> = EnhVec::new_with_capacity(1000);
        (0..=HEAD_SIZE as u32).for_each(|x: u32| ev.push_front(x));
        ev.sort(Sorting::Ascending); // folds the head into main
        assert!(ev.data.head.is_empty(), "head was compacted");
        let capacity: usize = ev.data.main.capacity();
        assert!(capacity >= 1000, "main capacity {capacity} kept");

        let mut ev: EnhVec<u32> = EnhVec::from_iter(0..1000);
        (0..1000).for_each(|x: u32| ev.push_front(x));
        let capacity: usize = ev.into_vec().capacity();
        assert!(capacity < 2100, "into_vec() capacity {capacity} not doubled");
    }

    #[test]
    #[rustfmt::skip]
    fn test_pop_both_ends_keeps_order() {
        let n: u32 = 1000;
        let mut ev: EnhVec<u32> = EnhVec::from_iter(0..n);
        let popped: Vec<u32> = from_fn(|| ev.pop_front()).collect();
        assert_eq!(popped, (0..n).collect::<Vec<u32>>(), "pop_front() after push()");

        (0..n).for_each(|x: u32| ev.push_front(x));
        let popped: Vec<u32> = from_fn(|| ev.pop()).collect();
        assert_eq!(popped, (0..n).collect::<Vec<u32>>(), "pop() after push_front()");

        // alternate between the ends, which rebalances between head and main
        (0..n).for_each(|x: u32| ev.push(x));
        let mut expected: VecDeque<u32> = (0..n).collect();
        for i in 0..n {
            let (got, want) = match i % 3 {
                0 => (ev.pop_front(), expected.pop_front()),
                _ => (ev.pop(), expected.pop_back()),
            };
            assert_eq!(got, want, "alternating pops, step {i}");
        }
    }

    #[test]
    fn test_eq_ignores_layout() {
        let ev1: EnhVec<u32> = EnhVec::from_iter([1, 2, 3]);
        let mut ev2: EnhVec<u32> = EnhVec::from_iter([2, 3]);
        ev2.push_front(1);
        assert_eq!(ev1, ev2, "same elements, different head/main split");

        let set: HashSet<EnhVec<u32>> = HashSet::from([ev1.clone()]);
        assert!(set.contains(&ev2), "usable as a HashSet key");

        ev2.push_front(0);
        assert_ne!(ev1, ev2, "different elements");
        let ev3: EnhVec<u32> = EnhVec::from_iter([3, 2, 1]);
        assert_ne!(ev1, ev3, "different order");
    }

    #[test]
    #[rustfmt::skip]
    fn test_hash() {
        let ev_asc: EnhVec<u32> = EnhVec::from_iter(PI_ASC);
        let ev_arr: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(std_hash(&ev_asc), std_hash(&ev_arr), "Hash ignores order");
        let pair1: (EnhVec<u32>, EnhVec<u32>) = (EnhVec::from_iter([1, 2]), EnhVec::from_iter([3]));
        let pair2: (EnhVec<u32>, EnhVec<u32>) = (EnhVec::from_iter([1]), EnhVec::from_iter([2, 3]));
        assert_ne!(std_hash(&pair1), std_hash(&pair2), "Hash of adjacent EnhVecs");

        let xxh_asc: EnhVec<XxhU32> = xxh3_vec(&PI_ASC);
        assert_eq!(xxh_asc.xxh3_digest(), xxh3_vec(&PI_ARR).xxh3_digest(), "xxh3 ignores order");
        let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
        xxh_asc.xxh3(&mut hasher);
        assert_eq!(hasher.finish(), xxh_asc.xxh3_digest(), "xxh3_digest() == xxh3() + finish()");

        let digest = |a: &[u32], b: &[u32]| {
            let mut hasher: CustomXxh3Hasher = CustomXxh3Hasher::default();
            xxh3_vec(a).xxh3(&mut hasher);
            xxh3_vec(b).xxh3(&mut hasher);
            hasher.finish()
        };
        assert_ne!(digest(&[1, 2], &[3]), digest(&[1], &[2, 3]), "xxh3 of adjacent EnhVecs");
    }

    #[test]
    #[rustfmt::skip]
    fn test_integer_overflow() {
        let ev: EnhVec<u8> = EnhVec::from_iter([200, 250]);
        assert_eq!(ev.average(), Some(225.0), "average u8");
        assert_eq!(ev.median(), Some(225), "median u8");
        let ev: EnhVec<i8> = EnhVec::from_iter([-128, 127]);
        assert_eq!(ev.median(), Some(0), "median i8, rounded towards zero");
        let ev: EnhVec<i32> = EnhVec::from_iter([-3, -2]);
        assert_eq!(ev.median(), Some(-2), "median i32, rounded towards zero");
        let ev: EnhVec<u64> = EnhVec::from_iter([u64::MAX, u64::MAX - 2]);
        assert_eq!(ev.median(), Some(u64::MAX - 1), "median u64");

        let ev: EnhVec<u32> = EnhVec::from_iter([10; 38]);
        assert_eq!(ev.product(), Some(10i128.pow(38)), "product");
        let ev: EnhVec<u32> = EnhVec::from_iter([10; 39]);
        assert_eq!(ev.product(), None, "product overflowing i128");
    }

    #[test]
    #[rustfmt::skip]
    fn test_variance_and_stdev_fp() {
        // same data and population variance as in test_variance_and_stdev()
        let ev: EnhVec<f64> = EnhVec::from_iter([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let var_diff: f64 = ev.variance_fp().unwrap() - 4.0;
        let std_diff: f64 = ev.stdev_fp().unwrap() - 2.0;
        assert!(var_diff.abs() < EPSILON, "variance diff ({var_diff}) not within epsilon");
        assert!(std_diff.abs() < EPSILON, "stdev diff ({std_diff}) not within epsilon");
    }

    #[test]
    fn test_percentile_nan() {
        let ev: EnhVec<u32> = EnhVec::from_iter(PI_ARR);
        assert_eq!(ev.percentile(f64::NAN), None, "percentile");
        let ev: EnhVec<f64> = EnhVec::from_iter(FP_ARR);
        assert_eq!(ev.percentile_fp(f64::NAN), None, "percentile_fp");
    }

    #[test]
    #[rustfmt::skip]
    fn test_fp_edge_values() {
        // total_cmp() orders (positive) NaN after all numbers
        let ev: EnhVec<f64> = EnhVec::from_iter([3.0, f64::NAN, 1.0, 2.0]);
        assert_eq!(ev.median_fp(), Some(2.5), "median_fp with NaN");
        assert_eq!(ev.percentile_fp(0.0), Some(1.0), "percentile_fp 0.0 with NaN");
        assert!(ev.percentile_fp(1.0).is_some_and(f64::is_nan), "percentile_fp 1.0 with NaN");

        let ev: EnhVec<f64> = EnhVec::from_iter([f64::MAX, f64::MAX]);
        assert_eq!(ev.median_fp(), Some(f64::MAX), "median_fp without overflow");
    }
}
