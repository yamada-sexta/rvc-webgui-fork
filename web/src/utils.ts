// export function getEnv(key: string, fallback: string): string {
//     const stored = localStorage.getItem(key);
//     if (stored === null || stored === undefined) {
//         localStorage.setItem(key, fallback);
//         return fallback;
//     }
//     return stored as string;
// }

// export function getFloat(key: string, fallback: number) {
//     return Number.parseFloat(getEnv(key, fallback.toString()))
// }

// export function getInt(key: string, fallback: number) {
//     return Number.parseInt(getEnv(key, fallback.toString()))
// }

export function getExtension(name: string) {
    const parts = name.split(".");
    return parts[parts.length - 1]
}