import Link from "next/link";
import { Home, PanelLeft } from "lucide-react";
import Image from "next/image";
import { useRouter } from "next/router";

import { Button } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTrigger } from "~/components/ui/sheet";
import { Avatar, AvatarFallback, AvatarImage } from "~/components/ui/avatar";
import { signOut, useSession } from "next-auth/react";

export function Header() {
  const { data: sessionData } = useSession();
  const router = useRouter();

  return (
    <header className="sticky top-0 z-30 flex h-24 items-center gap-4 border-b-2 border-primary bg-white px-6 pb-1">
      {/* Logo and Branding */}
      <div className="flex items-center space-x-3">
        <Link href="/">
          <Image
            src="/logoGS.png"
            alt="Santander Digital Seguro"
            width={170}
            height={50}
            className="cursor-pointer rounded-lg"
          />
        </Link>
        <div className="flex flex-col">
          <span className="text-xl font-bold text-gray-900">
            Santander Digital Seguro
          </span>
          <span className="text-xs text-muted-foreground">SDS</span>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="ml-8 hidden items-center space-x-6 md:flex">
        <Link
          href="/"
          className={`text-lg font-medium transition-colors hover:text-primary ${
            router.pathname === "/"
              ? "font-semibold text-primary"
              : "text-muted-foreground"
          }`}
        >
          Inicio
        </Link>
        <Link
          href="/chatbot"
          className={`text-lg font-medium transition-colors hover:text-primary ${
            router.pathname === "/chatbot"
              ? "font-semibold text-primary"
              : "text-muted-foreground"
          }`}
        >
          Gabi
        </Link>
        <Link
          href="/contact"
          className={`text-lg font-medium transition-colors hover:text-primary ${
            router.pathname === "/contact"
              ? "font-semibold text-primary"
              : "text-muted-foreground"
          }`}
        >
          Contáctenos
        </Link>
      </nav>

      {/* Mobile Menu */}
      <Sheet>
        <SheetTrigger asChild>
          <Button size="icon" variant="outline" className="md:hidden">
            <PanelLeft className="h-5 w-5" />
            <span className="sr-only">Toggle Menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="sm:max-w-xs">
          <nav className="grid gap-6 text-lg font-medium">
            <div className="mb-6 flex items-center space-x-3">
              <Image
                src="/logoGS.png"
                alt="Santander Digital Seguro"
                width={40}
                height={40}
                className="rounded-lg"
              />
              <div className="flex flex-col">
                <span className="text-lg font-bold text-gray-900">
                  Santander Digital Seguro
                </span>
                <span className="text-xs text-muted-foreground">SDS</span>
              </div>
            </div>
            <Link
              href="/"
              className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
            >
              <Home className="h-5 w-5" />
              Inicio
            </Link>
            <Link
              href="/chatbot"
              className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
            >
              <Home className="h-5 w-5" />
              Gabi
            </Link>
            <Link
              href="/contact"
              className="flex items-center gap-4 px-2.5 text-muted-foreground hover:text-foreground"
            >
              <Home className="h-5 w-5" />
              Contáctenos
            </Link>
          </nav>
        </SheetContent>
      </Sheet>

      <div className="ml-auto flex items-center space-x-4">
        {sessionData ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="overflow-hidden rounded-full"
              >
                <Avatar className="h-8 w-8">
                  <AvatarImage
                    src={sessionData?.user.image || ""}
                    alt="@shadcn"
                  />
                  <AvatarFallback>SC</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">
                    {sessionData?.user.name}
                  </p>
                  <p className="text-xs leading-none text-muted-foreground">
                    {sessionData?.user.email}
                  </p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => void signOut({ callbackUrl: "/" })}
              >
                Salir
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button
            variant="default"
            onClick={() => void router.push("/authentication")}
          >
            Iniciar sesión
          </Button>
        )}
      </div>
    </header>
  );
}
