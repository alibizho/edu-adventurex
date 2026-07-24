import { useNavigate } from "react-router-dom";
import { ROUTES } from "../../app/routes";
import { AppHeader } from "../../components/layout/AppHeader";
import { StatusBar } from "../../components/layout/StatusBar";

export function NotFoundPage() {
  const navigate = useNavigate();
  return (
    <div className="screen not-found-screen halftone-screen">
      <AppHeader />
      <main className="not-found-panel retro-panel">
        <span>ERROR CODE</span>
        <h1>404</h1>
        <p>PAGE NOT FOUND. THE REQUESTED MODULE DOES NOT EXIST.</p>
        <button type="button" className="solid-action" onClick={() => navigate(ROUTES.material)}>
          RETURN HOME
        </button>
      </main>
      <StatusBar label="ROUTE_NOT_FOUND" full meta="CPU: 00%   MEM: 000KB" />
    </div>
  );
}
