import Login from "./components/login";
import ThemeToggle from "./components/theme-toggle";

export default function Home() {
  return (
    <div className="min-h-screen flex items-center justify-center transition-colors">
      <main>
        <div className="absolute top-4 right-4">
          <ThemeToggle />
        </div>
        <Login />
      </main>
    </div>
  );
}
