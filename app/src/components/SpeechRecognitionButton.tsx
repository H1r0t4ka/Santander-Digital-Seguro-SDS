"use client";

import { useEffect, useImperativeHandle, forwardRef } from "react";
import { Button } from "~/components/ui/button";
import { Mic } from "lucide-react";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";

interface SpeechRecognitionButtonProps {
  onTranscriptChange: (transcript: string) => void;
  disabled?: boolean;
  onListeningChange?: (listening: boolean) => void;
}

export interface SpeechRecognitionButtonRef {
  stopListening: () => void;
}

const SpeechRecognitionButton = forwardRef<
  SpeechRecognitionButtonRef,
  SpeechRecognitionButtonProps
>(({ onTranscriptChange, disabled = false, onListeningChange }, ref) => {
  const {
    transcript,
    listening,
    browserSupportsSpeechRecognition,
    resetTranscript,
  } = useSpeechRecognition();

  // Update parent component with transcript changes
  useEffect(() => {
    onTranscriptChange(transcript);
  }, [transcript, onTranscriptChange]);

  // Update parent component with listening state changes
  useEffect(() => {
    if (onListeningChange) {
      onListeningChange(listening);
    }
  }, [listening, onListeningChange]);

  // Expose stopListening function to parent component
  useImperativeHandle(
    ref,
    () => ({
      stopListening: () => {
        if (listening) {
          SpeechRecognition.stopListening().catch(() => {
            // Handle error silently
          });
        }
      },
    }),
    [listening]
  );

  if (!browserSupportsSpeechRecognition) {
    return null;
  }

  const handleMicClick = () => {
    if (listening) {
      SpeechRecognition.stopListening().catch(() => {
        // Handle error silently
      });
    } else {
      // Reset transcript before starting new recording
      resetTranscript();
      SpeechRecognition.startListening({
        language: "es-ES",
        continuous: true,
      }).catch(() => {
        // Handle error silently
      });
    }
  };

  return (
    <Button
      variant="outline"
      size="default"
      className="h-10 px-3"
      onClick={handleMicClick}
      disabled={disabled}
      type="button"
    >
      <Mic className={`h-4 w-4 ${listening ? "text-red-500" : ""}`} />
    </Button>
  );
});

SpeechRecognitionButton.displayName = "SpeechRecognitionButton";

export default SpeechRecognitionButton;
