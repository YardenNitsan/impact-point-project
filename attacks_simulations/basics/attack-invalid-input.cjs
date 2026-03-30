const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function run() {
  const cases = [
    {
      name: "missing initialSpeed",
      body: {
        lat: 32.1,
        lon: 34.8,
        alt: 1000,
        azimuth: 120,
        elevation: 30,
        mass: 50,
        weather_source: "machine",
      },
    },
    {
      name: "wrong type",
      body: {
        lat: "not-a-number",
        lon: 34.8,
        alt: 1000,
        azimuth: 120,
        elevation: 30,
        mass: 50,
        initialSpeed: 300,
        weather_source: "machine",
      },
    },
    {
      name: "out of range",
      body: {
        lat: 999,
        lon: 34.8,
        alt: 1000,
        azimuth: 120,
        elevation: 30,
        mass: 50,
        initialSpeed: 300,
        weather_source: "machine",
      },
    },
    {
      name: "unexpected extra field",
      body: {
        lat: 32.1,
        lon: 34.8,
        alt: 1000,
        azimuth: 120,
        elevation: 30,
        mass: 50,
        initialSpeed: 300,
        weather_source: "machine",
        evil: "should-not-pass",
      },
    },
    {
      name: "invalid enum",
      body: {
        lat: 32.1,
        lon: 34.8,
        alt: 1000,
        azimuth: 120,
        elevation: 30,
        mass: 50,
        initialSpeed: 300,
        weather_source: "hacker-mode",
      },
    },
  ];

  for (const testCase of cases) {
    const res = await fetch(`${BASE_URL}/api/simulation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(testCase.body),
    });

    const text = await res.text();

    console.log(`\n=== ${testCase.name} ===`);
    console.log("status:", res.status);
    console.log("body:", text);
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
