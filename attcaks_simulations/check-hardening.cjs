const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

async function testOrigin(origin) {
  const res = await fetch(`${BASE_URL}/health`, {
    headers: {
      Origin: origin,
    },
  });

  console.log(`\nOrigin: ${origin}`);
  console.log("status:", res.status);
  console.log(
    "access-control-allow-origin:",
    res.headers.get("access-control-allow-origin"),
  );
  console.log("x-frame-options:", res.headers.get("x-frame-options"));
  console.log(
    "x-content-type-options:",
    res.headers.get("x-content-type-options"),
  );
  console.log(
    "content-security-policy:",
    res.headers.get("content-security-policy"),
  );
}

async function run() {
  await testOrigin("http://localhost:4200");
  await testOrigin("http://evil.example.com");
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
