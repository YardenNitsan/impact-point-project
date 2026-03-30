export const environmentService = {
  ALLOWED_ORIGIN: process.env.ALLOWED_ORIGIN ?? "http://localhost:4200",
  NODE_PORT: Number(process.env.NODE_PORT ?? 3000),
  MONGO_URI: process.env.MONGO_URI ?? "mongodb://mongo:27017/impact-point",
  PYTHON_SERVICE_URI:
    process.env.PYTHON_SERVICE_URI ?? "http://algorithm:8000/simulate-impact",
};
