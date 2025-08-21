import "./global.css"; // <-- REQUIRED

export const metadata = { title: "Syllabus RAG Assistant" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
