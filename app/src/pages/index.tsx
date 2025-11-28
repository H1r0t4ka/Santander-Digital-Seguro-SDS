import { useState, useEffect } from "react";
import { Header } from "~/components/header";
import { Progress } from "~/components/ui/progress";
import FloatingChatbotButton from "~/components/FloatingChatbotButton";

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isLoading) {
      const timer = setTimeout(() => {
        setProgress((oldProgress) => {
          if (oldProgress === 100) {
            return 100;
          }
          const diff = Math.random() * 10;
          return Math.min(oldProgress + diff, 100);
        });
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [isLoading, progress]);

  const handleIframeLoad = () => {
    setIsLoading(false);
    setProgress(100);
  };

  return (
    <>
      <Header />
      <main className="flex h-screen flex-col items-center justify-center">
        {isLoading && (
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="text-2xl font-semibold">
              Cargando Tablero Predictivo
            </div>
            <Progress value={progress} className="w-[80%]" />
            <div className="text-sm text-muted-foreground">
              {progress.toFixed(0)}% completado
            </div>
          </div>
        )}
        <iframe
          src="https://h1r0t4ka-sds.hf.space"
          onLoad={handleIframeLoad}
          style={{
            display: isLoading ? "none" : "block",
            width: "100%",
            height: "100%",
            border: "none",
          }}
        />
        {/* Floating Chatbot Button */}
        <FloatingChatbotButton />
      </main>
    </>
  );
}
