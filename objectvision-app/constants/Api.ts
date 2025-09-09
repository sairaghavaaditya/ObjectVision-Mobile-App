import Constants from 'expo-constants';

interface ExtraConfig {
  API_BASE?: string;
}

const extra = (Constants.expoConfig?.extra ?? {}) as ExtraConfig;

// IMPORTANT: On a real device with Expo Go, 'localhost' points to the phone.
// Set app.json > extra.API_BASE to your machine's LAN IP, e.g. "http://192.168.1.50:8000"
export const API_BASE = extra.API_BASE || 'http://localhost:8000';

export function getMimeFromUri(uri: string): string {
  const lowered = uri.toLowerCase();
  if (lowered.endsWith('.png')) return 'image/png';
  if (lowered.endsWith('.jpg') || lowered.endsWith('.jpeg')) return 'image/jpeg';
  if (lowered.endsWith('.bmp')) return 'image/bmp';
  if (lowered.endsWith('.tiff') || lowered.endsWith('.tif')) return 'image/tiff';
  if (lowered.endsWith('.webp')) return 'image/webp';
  // Fallback; many pickers return jpg
  return 'image/jpeg';
}



