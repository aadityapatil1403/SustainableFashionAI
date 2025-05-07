# Pattern Optimizer Demo Site

This is the frontend application for the CSCI 566 Final Project. It provides a user interface for interacting with the pattern placement algorithms.

1. **No-Fit Polygon (NFP) Optimizer** - Uses advanced No-Fit Polygon techniques to find optimal placements with rotation options.
2. **Hybrid Pattern Optimizer** - Combines initial bin packing with polygon refinement and includes seam allowances.
3. **Bin Packing Optimizer** - Uses a simple rectangle-based approach for fast and efficient packing.

## Features

- Select between three different optimization methods
- Upload pattern files (hoodie-pattern.pdf, hoodie2-pattern.pdf, hoodie3-pattern.pdf)
- View optimization results with metrics and visualizations
- Processing with a progress bar 

## Getting Started

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

