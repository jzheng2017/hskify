//! Process-wide admission for bounded browser CUDA inference phases.
//!
//! A permit covers exactly one detector tile batch, OCR batch, translation
//! batch, or translation repair batch. Releasing between those boundaries lets
//! visible work from another job overtake queued offscreen work.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use tokio::sync::Notify;

const CUDA_QUEUE_CAPACITY: usize = 64;
const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);

static GLOBAL_CUDA_SCHEDULER: OnceLock<Arc<CudaScheduler>> = OnceLock::new();

pub(crate) fn global_cuda_scheduler() -> Arc<CudaScheduler> {
    GLOBAL_CUDA_SCHEDULER
        .get_or_init(|| Arc::new(CudaScheduler::new(CUDA_QUEUE_CAPACITY)))
        .clone()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CudaPriority {
    Offscreen,
    Visible,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum CudaAdmissionError {
    #[error("CUDA inference was cancelled while queued")]
    Cancelled,
    #[error("CUDA inference queue is full (maximum {capacity} waiting phases)")]
    QueueFull { capacity: usize },
}

#[derive(Debug)]
pub(crate) struct CudaScheduler {
    capacity: usize,
    state: Mutex<SchedulerState>,
    changed: Notify,
}

#[derive(Debug, Default)]
struct SchedulerState {
    active: bool,
    next_sequence: u64,
    waiters: Vec<Waiter>,
}

#[derive(Debug)]
struct Waiter {
    sequence: u64,
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
}

impl SchedulerState {
    fn remove_cancelled(&mut self) {
        self.waiters
            .retain(|waiter| !waiter.cancel.load(Ordering::Acquire));
    }

    fn next_waiter_sequence(&self) -> Option<u64> {
        self.waiters
            .iter()
            .max_by(|left, right| {
                left.priority.cmp(&right.priority).then_with(|| {
                    // Earlier sequence numbers win within the same priority.
                    right.sequence.cmp(&left.sequence)
                })
            })
            .map(|waiter| waiter.sequence)
    }

    fn remove_waiter(&mut self, sequence: u64) -> bool {
        let before = self.waiters.len();
        self.waiters.retain(|waiter| waiter.sequence != sequence);
        self.waiters.len() != before
    }
}

impl CudaScheduler {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "CUDA queue capacity must be positive");
        Self {
            capacity,
            state: Mutex::new(SchedulerState::default()),
            changed: Notify::new(),
        }
    }

    pub(crate) async fn acquire(
        self: &Arc<Self>,
        priority: CudaPriority,
        cancel: Arc<AtomicBool>,
    ) -> Result<CudaPermit, CudaAdmissionError> {
        if cancel.load(Ordering::Acquire) {
            return Err(CudaAdmissionError::Cancelled);
        }

        let sequence = {
            let mut state = self.state.lock().expect("CUDA scheduler lock poisoned");
            state.remove_cancelled();
            if !state.active && state.waiters.is_empty() {
                state.active = true;
                return Ok(CudaPermit {
                    scheduler: self.clone(),
                });
            }
            if state.waiters.len() >= self.capacity {
                return Err(CudaAdmissionError::QueueFull {
                    capacity: self.capacity,
                });
            }
            let sequence = state.next_sequence;
            state.next_sequence = state.next_sequence.wrapping_add(1);
            state.waiters.push(Waiter {
                sequence,
                priority,
                cancel: cancel.clone(),
            });
            sequence
        };

        loop {
            if cancel.load(Ordering::Acquire) {
                self.cancel_waiter(sequence);
                return Err(CudaAdmissionError::Cancelled);
            }

            // Register before checking state so a release cannot be missed
            // between the state check and awaiting the notification.
            let changed = self.changed.notified();
            {
                let mut state = self.state.lock().expect("CUDA scheduler lock poisoned");
                state.remove_cancelled();
                if !state.active && state.next_waiter_sequence() == Some(sequence) {
                    let removed = state.remove_waiter(sequence);
                    debug_assert!(removed, "admitted CUDA waiter must still be queued");
                    state.active = true;
                    return Ok(CudaPermit {
                        scheduler: self.clone(),
                    });
                }
                if !state
                    .waiters
                    .iter()
                    .any(|waiter| waiter.sequence == sequence)
                {
                    return Err(CudaAdmissionError::Cancelled);
                }
            }

            tokio::select! {
                _ = changed => {}
                _ = tokio::time::sleep(CANCELLATION_POLL_INTERVAL) => {}
            }
        }
    }

    fn cancel_waiter(&self, sequence: u64) {
        let removed = self
            .state
            .lock()
            .expect("CUDA scheduler lock poisoned")
            .remove_waiter(sequence);
        if removed {
            self.changed.notify_waiters();
        }
    }

    fn release(&self) {
        {
            let mut state = self.state.lock().expect("CUDA scheduler lock poisoned");
            debug_assert!(state.active, "CUDA permit released without an active phase");
            state.active = false;
            state.remove_cancelled();
        }
        self.changed.notify_waiters();
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.state
            .lock()
            .expect("CUDA scheduler lock poisoned")
            .waiters
            .len()
    }
}

#[derive(Debug)]
pub(crate) struct CudaPermit {
    scheduler: Arc<CudaScheduler>,
}

impl Drop for CudaPermit {
    fn drop(&mut self) {
        self.scheduler.release();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    async fn wait_for_pending(scheduler: &CudaScheduler, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while scheduler.pending() != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scheduler did not reach the expected queue depth");
    }

    #[tokio::test]
    async fn visible_waiter_overtakes_queued_offscreen_waiter() {
        let scheduler = Arc::new(CudaScheduler::new(4));
        let active = scheduler
            .acquire(CudaPriority::Offscreen, Arc::new(AtomicBool::new(false)))
            .await
            .unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));

        let offscreen = {
            let scheduler = scheduler.clone();
            let order = order.clone();
            tokio::spawn(async move {
                let _permit = scheduler
                    .acquire(CudaPriority::Offscreen, Arc::new(AtomicBool::new(false)))
                    .await
                    .unwrap();
                order.lock().unwrap().push("offscreen");
            })
        };
        wait_for_pending(&scheduler, 1).await;
        let visible = {
            let scheduler = scheduler.clone();
            let order = order.clone();
            tokio::spawn(async move {
                let _permit = scheduler
                    .acquire(CudaPriority::Visible, Arc::new(AtomicBool::new(false)))
                    .await
                    .unwrap();
                order.lock().unwrap().push("visible");
            })
        };
        wait_for_pending(&scheduler, 2).await;

        drop(active);
        visible.await.unwrap();
        offscreen.await.unwrap();
        assert_eq!(*order.lock().unwrap(), ["visible", "offscreen"]);
    }

    #[tokio::test]
    async fn bounded_admission_fails_clearly() {
        let scheduler = Arc::new(CudaScheduler::new(1));
        let active = scheduler
            .acquire(CudaPriority::Offscreen, Arc::new(AtomicBool::new(false)))
            .await
            .unwrap();
        let queued = {
            let scheduler = scheduler.clone();
            tokio::spawn(async move {
                scheduler
                    .acquire(CudaPriority::Offscreen, Arc::new(AtomicBool::new(false)))
                    .await
            })
        };
        wait_for_pending(&scheduler, 1).await;

        let error = scheduler
            .acquire(CudaPriority::Visible, Arc::new(AtomicBool::new(false)))
            .await
            .unwrap_err();
        assert_eq!(error, CudaAdmissionError::QueueFull { capacity: 1 });

        drop(active);
        drop(queued.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn cancellation_is_observed_while_queued() {
        let scheduler = Arc::new(CudaScheduler::new(2));
        let active = scheduler
            .acquire(CudaPriority::Offscreen, Arc::new(AtomicBool::new(false)))
            .await
            .unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let queued = {
            let scheduler = scheduler.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move { scheduler.acquire(CudaPriority::Visible, cancel).await })
        };
        wait_for_pending(&scheduler, 1).await;
        cancel.store(true, Ordering::Release);

        let error = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .expect("queued cancellation was not observed")
            .unwrap()
            .unwrap_err();
        assert_eq!(error, CudaAdmissionError::Cancelled);
        assert_eq!(scheduler.pending(), 0);
        drop(active);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_one_cuda_phase_is_active() {
        let scheduler = Arc::new(CudaScheduler::new(16));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for index in 0..12 {
            let scheduler = scheduler.clone();
            let active = active.clone();
            let maximum = maximum.clone();
            tasks.push(tokio::spawn(async move {
                let _permit = scheduler
                    .acquire(
                        if index % 3 == 0 {
                            CudaPriority::Visible
                        } else {
                            CudaPriority::Offscreen
                        },
                        Arc::new(AtomicBool::new(false)),
                    )
                    .await
                    .unwrap();
                let now = active.fetch_add(1, Ordering::AcqRel) + 1;
                maximum.fetch_max(now, Ordering::AcqRel);
                tokio::task::yield_now().await;
                active.fetch_sub(1, Ordering::AcqRel);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(maximum.load(Ordering::Acquire), 1);
    }
}
