const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

const cases = [
  {
    name: "NoSQL-style object in lat + extra admin flag",
    body: {
      mass: 500,
      initialSpeed: 900,
      elevation: 35,
      azimuth: 35,
      lat: { $gt: 0 },
      lon: 35.2,
      alt: 19000,
      weather_source: "calculations",
      admin: true,
    },
  },
  {
    name: "Array and string type confusion",
    body: {
      mass: [500],
      initialSpeed: "900",
      elevation: 35,
      azimuth: 35,
      lat: 31.8,
      lon: 35.2,
      alt: 19000,
      weather_source: "calculations",
    },
  },
  {
    name: "Invalid enum and junk key",
    body: {
      mass: 500,
      initialSpeed: 900,
      elevation: 35,
      azimuth: 35,
      lat: 31.8,
      lon: 35.2,
      alt: 19000,
      weather_source: "evil",
      root: true,
    },
  },
];

async function run() {
  for (const testCase of cases) {
    const res = await fetch(`${BASE_URL}/api/simulation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(testCase.body),
    });

    console.log(`\n=== ${testCase.name} ===`);
    console.log("status:", res.status);
    console.log("body:", await res.text());
    console.log(
      res.status === 400 ? "RESULT: PROTECTED" : "RESULT: NOT BLOCKED",
    );
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
