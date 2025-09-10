import Constants from 'expo-constants';
import { Platform } from 'react-native';

interface ExtraConfig {
  API_BASE?: string;
}

const extra = (Constants.expoConfig?.extra ?? {}) as ExtraConfig;

function inferApiBase(): string {
  if (Platform.OS === 'web') {
    const host = typeof window !== 'undefined' && (window as any).location ? (window as any).location.hostname : 'localhost';
    const normalizedHost = host === 'localhost' ? '127.0.0.1' : host;
    return `http://${normalizedHost}:8000`;
  }
  if (extra.API_BASE) return extra.API_BASE;
  
  // For mobile devices, use your computer's IP address instead of localhost
  // Your computer's IP address is: 192.168.29.242
  // Make sure your mobile device and computer are on the same WiFi network
  return 'http://192.168.29.242:8000';
}

export const API_BASE = inferApiBase();

export function getMimeFromUri(uri: string): string {
  const lowered = uri.toLowerCase();
  if (lowered.endsWith('.png')) return 'image/png';
  if (lowered.endsWith('.jpg') || lowered.endsWith('.jpeg')) return 'image/jpeg';
  if (lowered.endsWith('.bmp')) return 'image/bmp';
  if (lowered.endsWith('.tiff') || lowered.endsWith('.tif')) return 'image/tiff';
  if (lowered.endsWith('.webp')) return 'image/webp';
  return 'image/jpeg';
}



