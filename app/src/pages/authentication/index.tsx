"use client";

import * as React from "react";
import { cn } from "~/lib/utils";
import { Icons } from "~/components/ui/icons";
import { Button } from "~/components/ui/button";
import { signIn, signOut, useSession } from "next-auth/react";
import { useEffect } from "react";
import { useRouter } from "next/router";
import Link from "next/link";

type UserAuthFormProps = React.HTMLAttributes<HTMLDivElement>;

export default function AuthenticationPage({
  className,
  ...props
}: UserAuthFormProps) {
  const { data: sessionData, status } = useSession();
  const router = useRouter();

  const handleGoogleLogin = async () => {
    await signIn("google", { callbackUrl: "/" });
  };

  const handleMicrosoftLogin = async () => {
    await signIn(
      "azure-ad",
      { callbackUrl: "/dashboard" },
      { prompt: "login" }
    );
  };

  useEffect(() => {
    if (status === "authenticated") {
      if (sessionData?.user?.role !== "USER") {
        void router.push("/dashboard");
      } else {
        void router.push("/unauthorized");
      }
    }
  }, [sessionData, router, status]);

  const allowedStatus = ["authenticated", "loading"];

  if (!allowedStatus.includes(status)) {
    return (
      <>
        <div className="container relative h-[1000px] flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
          <div className="relative hidden h-full flex-col bg-muted p-10 text-white dark:border-r lg:flex">
            <div className="absolute inset-0 bg-primary" />
            <div className="relative z-20 flex items-center text-lg font-medium">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mr-2 h-6 w-6"
              >
                <path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3" />
              </svg>
              Transitapp
            </div>
            <div className="relative z-20 mt-auto">
              <blockquote className="space-y-2">
                <p className="text-lg">
                  Asistente virtual para educar, orientar y prevenir sobre la
                  ley 769 2002 - código de tránsito.
                </p>
                <footer className="text-sm">By MarimondAI</footer>
              </blockquote>
            </div>
          </div>
          <div className="mt-8 lg:p-8">
            <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
              <div className="flex flex-col space-y-2 text-center">
                <h1 className="text-2xl font-semibold tracking-tight">
                  Bienvenido
                </h1>
              </div>

              <div className={cn("grid gap-6", className)} {...props}>
                <Button
                  variant="outline"
                  type="button"
                  onClick={
                    sessionData
                      ? () => void signOut()
                      : () => void handleMicrosoftLogin()
                  }
                >
                  <Icons.microsoft className="mr-2 h-4 w-4" />{" "}
                  {sessionData ? " Sign out" : " Sign in with Microsoft"}
                </Button>
                <Button
                  variant="outline"
                  type="button"
                  onClick={
                    sessionData
                      ? () => void signOut()
                      : () => void handleGoogleLogin()
                  }
                >
                  <Icons.google className="mr-2 h-4 w-4" />{" "}
                  {sessionData ? " Sign out" : " Sign in with Google"}
                </Button>
              </div>
              <p className="px-8 text-center text-sm text-muted-foreground">
                Al hacer clic en Continuar, acepta nuestros{" "}
                <Link
                  href="/terms"
                  className="underline underline-offset-4 hover:text-primary"
                >
                  Términos de servicio
                </Link>{" "}
                y{" "}
                <Link
                  href="/privacy"
                  className="underline underline-offset-4 hover:text-primary"
                >
                  Política de privacidad
                </Link>
                .
              </p>
            </div>
          </div>
        </div>
      </>
    );
  }
}