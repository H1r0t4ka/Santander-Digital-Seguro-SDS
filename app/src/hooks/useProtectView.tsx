import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { useEffect } from "react";

const useProtectView = (location: string) => {
  const { status } = useSession(); //data,
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") {
      void router.push(location);
    }
  }, [status, router, location]);
  return status;
};

export default useProtectView;
