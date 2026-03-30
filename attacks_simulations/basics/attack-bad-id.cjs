const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function test(method, path) {
  const res = await fetch(`${BASE_URL}${path}`, { method });
  const text = await res.text();

  console.log(`\n${method} ${path}`);
  console.log("status:", res.status);
  console.log("body:", text);
}

async function run() {
  const badIds = [
    "1",
    "abc",
    "not-a-real-object-id",
    "!!!!!!!!!!!!!!!!!!!!!!!!",
    "12345678901234567890123",
    "1234567890123456789012345",
  ];

  for (const id of badIds) {
    await test("GET", `/api/simulation/${id}`);
    await test("GET", `/api/simulation/${id}/details`);
    await test("DELETE", `/api/simulation/${id}`);
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
