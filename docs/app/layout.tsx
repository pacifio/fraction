import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter } from 'next/font/google';
import type { Metadata } from 'next';

const inter = Inter({
  subsets: ['latin'],
});

const siteUrl = 'https://fraction.dev';
const description =
  'Persistent memory layer for LLM agents. Zero API costs, sub-100ms ingestion, fully offline. Outperforms mem0 and supermemory.';

export const metadata: Metadata = {
  title: {
    default: 'Fraction',
    template: '%s | Fraction',
  },
  description,
  metadataBase: new URL(siteUrl),
  openGraph: {
    type: 'website',
    siteName: 'Fraction',
    title: 'Fraction',
    description,
    url: siteUrl,
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Fraction',
    description,
  },
  other: {
    'github:repo': 'https://github.com/pacifio/fraction',
    'pypi:package': 'https://pypi.org/project/fractionally/',
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
