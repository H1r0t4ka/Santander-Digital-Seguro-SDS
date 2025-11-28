import Link from "next/link";
import Image from "next/image";
import { Button } from "~/components/ui/button";

const FloatingChatbotButton = () => {
  return (
    <Link href="/chatbot">
      <Button
        variant="outline"
        className="fixed bottom-6 right-6 z-50 h-20 w-20 rounded-full border-primary p-0 shadow-lg transition-all hover:scale-110"
        size="icon"
      >
        <Image
          src="/logoGabi.png"
          alt="Chatbot"
          width={48}
          height={48}
          className="rounded-full"
        />
      </Button>
    </Link>
  );
};

export default FloatingChatbotButton;