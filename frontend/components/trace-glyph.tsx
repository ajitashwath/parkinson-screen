"use client";

import { motion, useReducedMotion } from "framer-motion";

export function TraceGlyph() {
  const reduceMotion = useReducedMotion();
  return (
    <svg aria-hidden="true" className="h-full w-full" fill="none" viewBox="0 0 600 600">
      <motion.circle cx="300" cy="300" opacity="0.4" r="216" stroke="#cef79e" strokeWidth="1" initial={reduceMotion ? false : { scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 0.4 }} transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }} />
      <motion.path d="M300 300c0-14 12-25 26-25 34 0 62 28 62 62 0 78-63 141-141 141-116 0-210-94-210-210 0-137 111-248 248-248 158 0 286 128 286 286" stroke="#cef79e" strokeLinecap="round" strokeWidth="2" initial={reduceMotion ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 2.2, ease: [0.22, 1, 0.36, 1] }} />
      <motion.circle cx="300" cy="300" fill="#0007cb" r="5" initial={reduceMotion ? false : { scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.7, duration: 0.4 }} />
      <path d="M96 504h408" opacity="0.35" stroke="#dedbd6" />
    </svg>
  );
}
