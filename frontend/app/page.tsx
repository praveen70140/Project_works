"use client";

import Image from "next/image";
import Login from "./components/login";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-primary-50">
      <Login />
    </div>
  );
}
