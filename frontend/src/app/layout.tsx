import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GPT-2 124M Chat",
  description:
    "Local ChatGPT-style interface powered by a custom PyTorch GPT-2 124M model",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-emerald-500/30 selection:text-emerald-100">
        {children}
      </body>
    </html>
  );
}

