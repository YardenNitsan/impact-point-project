import axios from "axios";

const MONGO_SERVICE_URL = "http://localhost:4000/api";

export const saveSimulation = async (data: any) => {
  const res = await axios.post(
    `${MONGO_SERVICE_URL}/simulation-input`,
    data
  );
  return res.data;
};

export const getSimulations = async () => {
  const res = await axios.get(
    `${MONGO_SERVICE_URL}/simulation-result`
  );
  return res.data;
};