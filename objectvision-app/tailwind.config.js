module.exports = {
	content: [
		"./app/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
	],
	theme: {
		extend: {
			colors: {
				primary: { DEFAULT: '#20B2AA', foreground: '#ffffff' },
				muted: '#f5f5f5',
			}
		}
	},
	plugins: [],
};

