import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { PixelTeacherAvatar } from "../visuals/PixelTeacherAvatar";

export function AppHeader() {
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  // Two ways in, and that is the whole model: start something new, or open the map and pick a
  // class off it. TEACH pointed at whichever class was last opened — a bookmark, not a place —
  // and it sat next to a PROGRESS page that showed a table of the same classes the map already
  // draws. Everything you can teach is reachable from the map.
  const navItems = [
    { label: "HOME", to: ROUTES.material, end: true },
    { label: "MAP", to: ROUTES.map, end: false },
  ] as const;

  useEffect(() => setIsMenuOpen(false), [location.pathname]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setIsMenuOpen(false);
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <header className="app-header">
      <NavLink className="brand" to={ROUTES.material} aria-label="WUT? home">WUT?</NavLink>
      <nav
        id="primary-navigation"
        className={`primary-nav ${isMenuOpen ? "is-open" : ""}`}
        aria-label="Primary navigation"
      >
        {navItems.map(({ label, to, end }) => (
          <NavLink
            key={label}
            to={to}
            end={end}
            className={({ isActive }) => isActive ? "is-active" : undefined}
            onClick={() => setIsMenuOpen(false)}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="header-tools">
        <button
          type="button"
          className="header-menu-button"
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={isMenuOpen}
          aria-controls="primary-navigation"
          onClick={() => setIsMenuOpen((current) => !current)}
        >
          {isMenuOpen ? <X size={29} strokeWidth={3} /> : <Menu size={29} strokeWidth={3} />}
        </button>
        <div className="teacher-profile"><PixelTeacherAvatar /></div>
      </div>
    </header>
  );
}
