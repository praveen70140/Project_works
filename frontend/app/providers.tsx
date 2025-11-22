"use client";

import { HeroUIProvider, ToastProvider } from "@heroui/react";
import { ThemeProvider } from "next-themes";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <HeroUIProvider locale="en-in">
      <ThemeProvider attribute="class" enableSystem>
        <ToastProvider />
        {children}
      </ThemeProvider>
    </HeroUIProvider>
  );
}
