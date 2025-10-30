/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx}', './public/index.html'],
  theme: {
    extend: {
      colors: {
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
        },
        orange: {
          400: '#FF6F3C',
          500: '#E85B27',
          600: '#C24916',
        },
        magenta: {
          400: '#D6336C',
          500: '#B32B5B',
          600: '#8A1F46',
        },
        slate: {
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
      },
      fontFamily: {
        mono: ['"Fira Code"', 'monospace'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        neonCyan:
          '0 0 5px #06b6d4, 0 0 15px #06b6d4, 0 0 20px #06b6d4, 0 0 40px #06b6d4',
        neonOrange:
          '0 0 5px #FF6F3C, 0 0 15px #FF6F3C, 0 0 20px #FF6F3C, 0 0 40px #FF6F3C',
        neonMagenta:
          '0 0 5px #D6336C, 0 0 15px #D6336C, 0 0 20px #D6336C, 0 0 40px #D6336C',
      },
      dropShadow: {
        cyan: '0 0 6px #06b6d4',
        orange: '0 0 6px #FF6F3C',
        magenta: '0 0 6px #D6336C',
      },
      animation: {
        glowCyan: 'glowCyan 2s infinite ease-in-out',
        glowOrange: 'glowOrange 2s infinite ease-in-out',
        glowMagenta: 'glowMagenta 2s infinite ease-in-out',
      },
      keyframes: {
        glowCyan: {
          '0%, 100%': { textShadow: '0 0 8px #06b6d4' },
          '50%': { textShadow: '0 0 20px #22d3ee' },
        },
        glowOrange: {
          '0%, 100%': { textShadow: '0 0 8px #FF6F3C' },
          '50%': { textShadow: '0 0 20px #FF6F3C' },
        },
        glowMagenta: {
          '0%, 100%': { textShadow: '0 0 8px #D6336C' },
          '50%': { textShadow: '0 0 20px #D6336C' },
        },
      },
      scrollbar: ['rounded'],
    },
  },
};
