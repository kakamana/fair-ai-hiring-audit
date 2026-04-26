import "./globals.css";

export const metadata = {
  title: "Fair AI Hiring",
  description: "Résumé screening with explicit fairness constraints + bias audit",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
