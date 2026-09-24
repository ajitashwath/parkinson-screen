"use client";

import { motion, useReducedMotion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1] } },
};

const staggerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};

export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduceMotion ? false : "hidden"}
      whileInView={reduceMotion ? undefined : "visible"}
      transition={{ delay: reduceMotion ? 0 : delay }}
      variants={revealVariants}
      viewport={{ once: true, amount: 0.18 }}
    >
      {children}
    </motion.div>
  );
}

export function Stagger({ children, className }: { children: ReactNode; className?: string }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduceMotion ? false : "hidden"}
      whileInView={reduceMotion ? undefined : "visible"}
      variants={staggerVariants}
      viewport={{ once: true, amount: 0.12 }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({ children, className }: { children: ReactNode; className?: string }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div className={className} variants={reduceMotion ? undefined : revealVariants}>
      {children}
    </motion.div>
  );
}

export function Rule({ className }: { className?: string }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      aria-hidden="true"
      className={`h-px origin-left bg-current ${className ?? ""}`}
      initial={reduceMotion ? false : { scaleX: 0 }}
      whileInView={reduceMotion ? undefined : { scaleX: 1 }}
      transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
      viewport={{ once: true, amount: 0.8 }}
    />
  );
}
