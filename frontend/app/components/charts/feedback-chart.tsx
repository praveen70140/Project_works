"use client";
import { feedbackApi, healthApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";

const fetcher = ([url, payload]: [url: string, payload: mainFormData]) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      Resturant_Name: payload.restaurantName,
      Cuisine: payload.cuisine,
      Location: payload.location,
      City: payload.city,
    }),
  }).then((r) => r.json());

const FeedbackChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<feedbackApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/feedback`, payload],
    fetcher
  );

  return (
    <Card>
      <CardBody>
        <p>Data: {JSON.stringify(data)}</p>
        <p>Error: {JSON.stringify(error)}</p>
        <p>Is Loading: {JSON.stringify(isLoading)}</p>
      </CardBody>
    </Card>
  );
};

export default FeedbackChart;
