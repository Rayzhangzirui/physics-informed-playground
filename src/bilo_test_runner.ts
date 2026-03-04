/**
 * Command-line test runner for BILO.
 * Run: npm run test-bilo
 */

declare const process: { exit: (code: number) => void };

import { runBiloTests } from "./bilo";

const result = runBiloTests();
result.messages.forEach(function(m) { console.log(m); });
console.log(result.pass ? "All tests passed." : "Some tests failed.");
process.exit(result.pass ? 0 : 1);
