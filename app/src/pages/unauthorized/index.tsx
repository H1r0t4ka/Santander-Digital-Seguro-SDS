import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert";
import { useSession } from "next-auth/react";

export default function Unauthorized() {
  useSession({ required: true });

  return (
    <div className="preview flex min-h-[350px] w-full items-center justify-center p-10">
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Alerta</AlertTitle>
        <AlertDescription>
          Comuníquese con el administrador para solicitar los permisos.
        </AlertDescription>
      </Alert>
    </div>
  );
}
