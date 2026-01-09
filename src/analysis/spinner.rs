// Released under MIT License.
// Copyright (c) 2024-2026 Ladislav Bartos

//! Implementation of a minimalistic spinner.

use std::io::{stdout, Write};

use colored::Colorize;

use crate::PANIC_MESSAGE;

pub(super) struct Spinner {
    symbols: &'static [char],
    index: usize,
    silent: bool,
}

impl Spinner {
    pub(super) fn new(silent: bool) -> Self {
        Spinner {
            symbols: &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            index: 0,
            silent,
        }
    }

    /// Advance the spinner one tick.
    pub(super) fn tick(&mut self, percentage: usize) {
        if !self.silent {
            let percentage = format!("{}%", percentage).yellow();
            print!(
                "    {} Processing particles [{}]\r",
                self.symbols[self.index], percentage
            );
            stdout().flush().expect(PANIC_MESSAGE);
            self.index = (self.index + 1) % self.symbols.len();
        }
    }

    pub(super) fn done(&self) {
        if !self.silent {
            let check = "✔".bright_green();
            let percentage = "100%".bright_green();
            println!("    {} Processing particles [{}]", check, percentage);
        }
    }
}
