// Copyright (c) 2026 Mikko Tanner. All rights reserved.

/*!
EnhVec compared to Vec and VecDeque on common operations, with `u64` elements.
Popped and read values are summed, so that the work cannot be optimized away.

Each group is one operation, with a benchmark per container and size, e.g.
`push_front/VecDeque/10000`. Cases that are quadratic for a container (e.g.
`Vec::insert(0, x)`) only run up to `QUADRATIC_MAX` elements. Run a subset
with e.g. `cargo bench --bench vs_std -- pop_front`.
*/

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use enhvec::{EnhVec, Sorting};
use std::{collections::VecDeque, iter::from_fn, mem, time::Duration};

const SIZES: [usize; 2] = [10_000, 1_000_000];
/// Largest size at which quadratic cases are run.
const QUADRATIC_MAX: usize = 10_000;
const VEC: &str = "Vec";
const DEQUE: &str = "VecDeque";
const ENHVEC: &str = "EnhVec";

/// Deterministic pseudo-random values (xorshift).
fn random(n: usize, seed: u64) -> Vec<u64> {
    let mut x: u64 = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x >> 20
        })
        .collect()
}

/// The same data in each container, built by alternating pushes to the back and front.
fn alternating_deque(data: &[u64]) -> VecDeque<u64> {
    let mut d: VecDeque<u64> = VecDeque::with_capacity(data.len());
    for (i, &x) in data.iter().enumerate() {
        match i % 2 {
            0 => d.push_back(x),
            _ => d.push_front(x),
        }
    }
    d
}

fn alternating_vec(data: &[u64]) -> Vec<u64> {
    Vec::from(alternating_deque(data))
}

fn alternating_enhvec(data: &[u64]) -> EnhVec<u64> {
    let mut e: EnhVec<u64> = EnhVec::new();
    for (i, &x) in data.iter().enumerate() {
        match i % 2 {
            0 => e.push(x),
            _ => e.push_front(x),
        }
    }
    e
}

fn sorted(data: &[u64]) -> Vec<u64> {
    let mut v: Vec<u64> = data.to_vec();
    v.sort();
    v
}

/// Benchmark `routine` on a fresh input from `setup`, which is not measured.
fn bench<S, R>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    n: usize,
    mut setup: impl FnMut() -> S,
    mut routine: impl FnMut(&mut S) -> R,
) {
    group.bench_function(BenchmarkId::new(name, n), |b| {
        b.iter_batched_ref(&mut setup, &mut routine, BatchSize::LargeInput);
    });
}

/// Run `f` in a new benchmark group for each size.
fn groups(c: &mut Criterion, name: &str, mut f: impl FnMut(&mut BenchmarkGroup<WallTime>, usize)) {
    let mut group: BenchmarkGroup<WallTime> = c.benchmark_group(name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for n in SIZES {
        f(&mut group, n);
    }
    group.finish();
}

/* ============================== pushes ============================== */

fn pushes(c: &mut Criterion) {
    groups(c, "push_back", |g, n| {
        let data: Vec<u64> = random(n, 42);
        bench(g, VEC, n, Vec::new, |v| {
            data.iter().for_each(|&x| v.push(x))
        });
        bench(g, DEQUE, n, VecDeque::new, |v| {
            data.iter().for_each(|&x| v.push_back(x))
        });
        bench(g, ENHVEC, n, EnhVec::new, |v| {
            data.iter().for_each(|&x| v.push(x))
        });
    });

    groups(c, "push_front", |g, n| {
        let data: Vec<u64> = random(n, 42);
        if n <= QUADRATIC_MAX {
            bench(g, VEC, n, Vec::new, |v| {
                data.iter().for_each(|&x| v.insert(0, x))
            });
        }
        bench(g, DEQUE, n, VecDeque::new, |v| {
            data.iter().for_each(|&x| v.push_front(x))
        });
        bench(g, ENHVEC, n, EnhVec::new, |v| {
            data.iter().for_each(|&x| v.push_front(x))
        });
        let swap: &str = "EnhVec_push_swap_front";
        bench(g, swap, n, EnhVec::new, |v| {
            data.iter().for_each(|&x| v.push_swap_front(x))
        });
    });

    groups(c, "push_both_ends", |g, n| {
        let data: Vec<u64> = random(n, 42);
        if n <= QUADRATIC_MAX {
            bench(g, VEC, n, || (), |_| alternating_vec(&data));
        }
        bench(g, DEQUE, n, || (), |_| alternating_deque(&data));
        bench(g, ENHVEC, n, || (), |_| alternating_enhvec(&data));
    });
}

/* =============================== pops =============================== */

fn pops(c: &mut Criterion) {
    groups(c, "pop_back", |g, n| {
        let data: Vec<u64> = random(n, 42);
        bench(
            g,
            VEC,
            n,
            || data.clone(),
            |v| from_fn(|| v.pop()).sum::<u64>(),
        );
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| from_fn(|| v.pop_back()).sum::<u64>(),
        );
        bench(
            g,
            ENHVEC,
            n,
            || EnhVec::from(data.clone()),
            |v| from_fn(|| v.pop()).sum::<u64>(),
        );
    });

    groups(c, "pop_front", |g, n| {
        let data: Vec<u64> = random(n, 42);
        if n <= QUADRATIC_MAX {
            bench(
                g,
                VEC,
                n,
                || data.clone(),
                |v| (0..v.len()).map(|_| v.remove(0)).sum::<u64>(),
            );
        }
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| from_fn(|| v.pop_front()).sum::<u64>(),
        );
        bench(
            g,
            ENHVEC,
            n,
            || EnhVec::from(data.clone()),
            |v| from_fn(|| v.pop_front()).sum::<u64>(),
        );
        let swap: &str = "EnhVec_swap_pop_front";
        bench(
            g,
            swap,
            n,
            || EnhVec::from(data.clone()),
            |v| from_fn(|| v.swap_pop_front()).sum::<u64>(),
        );
    });

    // a full queue: each new element pushed at one end, the oldest popped at the other
    groups(c, "queue_push_back_pop_front", |g, n| {
        let data: Vec<u64> = random(n, 42);
        if n <= QUADRATIC_MAX {
            bench(
                g,
                VEC,
                n,
                || data.clone(),
                |v| {
                    data.iter()
                        .map(|&x| {
                            v.push(x);
                            v.remove(0)
                        })
                        .sum::<u64>()
                },
            );
        }
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| {
                data.iter()
                    .map(|&x| {
                        v.push_back(x);
                        v.pop_front().unwrap()
                    })
                    .sum::<u64>()
            },
        );
        bench(
            g,
            ENHVEC,
            n,
            || EnhVec::from(data.clone()),
            |v| {
                data.iter()
                    .map(|&x| {
                        v.push(x);
                        v.pop_front().unwrap()
                    })
                    .sum::<u64>()
            },
        );
    });

    groups(c, "queue_push_front_pop_back", |g, n| {
        let data: Vec<u64> = random(n, 42);
        if n <= QUADRATIC_MAX {
            bench(
                g,
                VEC,
                n,
                || data.clone(),
                |v| {
                    data.iter()
                        .map(|&x| {
                            v.insert(0, x);
                            v.pop().unwrap()
                        })
                        .sum::<u64>()
                },
            );
        }
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| {
                data.iter()
                    .map(|&x| {
                        v.push_front(x);
                        v.pop_back().unwrap()
                    })
                    .sum::<u64>()
            },
        );
        bench(
            g,
            ENHVEC,
            n,
            || EnhVec::from(data.clone()),
            |v| {
                data.iter()
                    .map(|&x| {
                        v.push_front(x);
                        v.pop().unwrap()
                    })
                    .sum::<u64>()
            },
        );
    });
}

/* ============================== access ============================== */

// data built at both ends, so that EnhVec has elements in its head and main Vecs
fn access(c: &mut Criterion) {
    groups(c, "index_random", |g, n| {
        let data: Vec<u64> = random(n, 42);
        let idx: Vec<usize> = random(n, 7).into_iter().map(|x| x as usize % n).collect();
        bench(
            g,
            VEC,
            n,
            || alternating_vec(&data),
            |v| idx.iter().map(|&i| v[i]).sum::<u64>(),
        );
        bench(
            g,
            DEQUE,
            n,
            || alternating_deque(&data),
            |v| idx.iter().map(|&i| v[i]).sum::<u64>(),
        );
        bench(
            g,
            ENHVEC,
            n,
            || alternating_enhvec(&data),
            |v| idx.iter().map(|&i| v[i]).sum::<u64>(),
        );
    });

    groups(c, "iter_sum", |g, n| {
        let data: Vec<u64> = random(n, 42);
        bench(
            g,
            VEC,
            n,
            || alternating_vec(&data),
            |v| v.iter().sum::<u64>(),
        );
        bench(
            g,
            DEQUE,
            n,
            || alternating_deque(&data),
            |v| v.iter().sum::<u64>(),
        );
        bench(
            g,
            ENHVEC,
            n,
            || alternating_enhvec(&data),
            |v| v.iter().sum::<u64>(),
        );
    });

    groups(c, "into_vec", |g, n| {
        let data: Vec<u64> = random(n, 42);
        bench(g, VEC, n, || alternating_vec(&data), mem::take);
        bench(
            g,
            DEQUE,
            n,
            || alternating_deque(&data),
            |v| Vec::from(mem::take(v)),
        );
        bench(
            g,
            ENHVEC,
            n,
            || alternating_enhvec(&data),
            |v| mem::take(v).into_vec(),
        );
    });
}

/* ========================= sorting & stats ========================= */

fn sorting(c: &mut Criterion) {
    groups(c, "sort_random", |g, n| {
        let data: Vec<u64> = random(n, 42);
        bench(g, VEC, n, || alternating_vec(&data), |v| v.sort());
        bench(
            g,
            DEQUE,
            n,
            || alternating_deque(&data),
            |v| v.make_contiguous().sort(),
        );
        bench(
            g,
            ENHVEC,
            n,
            || alternating_enhvec(&data),
            |v| v.sort(Sorting::Ascending),
        );
    });

    groups(c, "sort_sorted", |g, n| {
        let data: Vec<u64> = sorted(&random(n, 42));
        let sorted_enhvec = || {
            let mut e: EnhVec<u64> = EnhVec::from(data.clone());
            e.sort(Sorting::Ascending);
            e
        };
        bench(g, VEC, n, || data.clone(), |v| v.sort());
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| v.make_contiguous().sort(),
        );
        bench(g, ENHVEC, n, sorted_enhvec, |v| v.sort(Sorting::Ascending));
    });

    groups(c, "insert_sorted", |g, n| {
        if n > QUADRATIC_MAX {
            return;
        }
        let data: Vec<u64> = sorted(&random(n, 42));
        let new: Vec<u64> = random(n, 99);
        let sorted_enhvec = || {
            let mut e: EnhVec<u64> = EnhVec::from(data.clone());
            e.sort(Sorting::Ascending);
            e
        };
        bench(
            g,
            VEC,
            n,
            || data.clone(),
            |v| {
                new.iter()
                    .for_each(|&x| v.insert(v.partition_point(|&y| y <= x), x))
            },
        );
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| {
                new.iter()
                    .for_each(|&x| v.insert(v.partition_point(|&y| y <= x), x))
            },
        );
        bench(g, ENHVEC, n, sorted_enhvec, |v| {
            new.iter().for_each(|&x| v.insert_sorted(x))
        });
    });

    // Vec and VecDeque copy the data first too, as EnhVec::median() takes &self
    groups(c, "median", |g, n| {
        let data: Vec<u64> = random(n, 42);
        let mid: usize = n / 2;
        bench(
            g,
            VEC,
            n,
            || data.clone(),
            |v| *v.clone().select_nth_unstable(mid).1,
        );
        bench(
            g,
            DEQUE,
            n,
            || VecDeque::from(data.clone()),
            |v| *v.clone().make_contiguous().select_nth_unstable(mid).1,
        );
        bench(g, ENHVEC, n, || EnhVec::from(data.clone()), |v| v.median());
    });
}

criterion_group!(benches, pushes, pops, access, sorting);
criterion_main!(benches);
