import axios from "axios";

const MONGO_SERVICE_URL = "http://localhost:4000/api";

export const saveSimulation = async (data: any) => {
  const res = await axios.post(
    `${MONGO_SERVICE_URL}/simulation`,
    data
  );
  return res.data;
};

export const getSimulations = async () => {
  const res = await axios.get(
    `${MONGO_SERVICE_URL}/simulation`
  );
  return res.data;
};

export const getSimulationToWatch = async (id: string ) =>{
  const res = (await axios.get(
    `${MONGO_SERVICE_URL}/simulation/${id}`
  ));
  return res.data
};

export const getSimulationDetails = async (id: string) => {
  const res = await axios.get(
    `${MONGO_SERVICE_URL}/simulation/${id}/details`
  );
  return res.data;
};


export const deleteSimulation = async (id: string) => {
  const res = await axios.delete(
    `${MONGO_SERVICE_URL}/simulation/${id}`
  );
  return res.data;
};