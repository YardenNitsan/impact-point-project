const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function run() {
  const body = {
    lat: 31.8,
    lon: 35.2,
    alt: 19000,
    azimuth: 35,
    elevation: 35,
    mass: 500,
    initialSpeed: 900,
    weather_source: "calculations",
    junk: "x".repeat(2 * 1024 * 1024),
  };

  const res = await fetch(`${BASE_URL}/api/simulation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  console.log("status:", res.status);
  console.log("body:", await res.text());
  console.log(res.status === 413 ? "RESULT: PROTECTED" : "RESULT: NOT BLOCKED");
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
