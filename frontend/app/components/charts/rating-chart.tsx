import { ratingApi, successProbabilityApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";

const fetcher = ([url, payload]: [url: string, payload: mainFormData]) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      year: payload.date.year,
      month: payload.date.month,
      sales_qty: payload.salesQuantity,
      sales_amount: payload.salesAmount,
      City: payload.city,
      Cuisine: payload.cuisine,
    }),
  }).then((r) => r.json());

const RatingChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<ratingApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/rf_rating`, payload],
    fetcher
  );

  if (isLoading)
    return (
      <div>
        <CircularProgress />
      </div>
    );
  if (!data) return <div>Not found</div>;

  return (
    <Card className="col-span-2">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start absolute top-3">Expected Rating</h3>

        <CircularProgress
          classNames={{
            svg: "w-36 h-36 drop-shadow-md",
            indicator: "",
            track: "",
            value: "text-3xl font-semibold ",
          }}
          color="warning"
          minValue={0}
          maxValue={5}
          showValueLabel={true}
          formatOptions={{ style: "decimal" }}
          label="Stars"
          strokeWidth={4}
          value={Number(data.rf_rating_prediction.toFixed(2))}
        />
      </CardBody>
    </Card>
  );
};

export default RatingChart;
