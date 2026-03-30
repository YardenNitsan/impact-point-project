const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function run() {
  const huge = "A".repeat(2 * 1024 * 1024);

  const body = {
    lat: 32.1,
    lon: 34.8,
    alt: 1000,
    azimuth: 120,
    elevation: 30,
    mass: 50,
    initialSpeed: 300,
    weather_source: "machine",
    hugeField: huge,
  };

  const res = await fetch(`${BASE_URL}/api/simulation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();

  console.log("status:", res.status);
  console.log("body:", text);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
