import { successProbabilityApi } from "@/app/types/api-types";
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
      Ratings: payload.rating,
      City: payload.city,
      Cuisine: payload.cuisine,
    }),
  }).then((r) => r.json());

const SuccessChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<successProbabilityApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/rf_success_prob`, payload],
    fetcher
  );

  if (isLoading)
    return (
      <div>
        <CircularProgress />
      </div>
    );
  if (!data) return <div>Not found</div>;

  const statusColor = data.is_successful
    ? "text-success-500"
    : "text-danger-500";

  return (
    <Card className="col-span-2">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start absolute top-3">Success Probability</h3>

        <CircularProgress
          classNames={{
            svg: "w-36 h-36 drop-shadow-md",
            indicator: "",
            track: "",
            value: "text-3xl font-semibold ",
          }}
          showValueLabel={true}
          strokeWidth={4}
          value={data.success_probability_percentage}
        />
        <p className="text-2xl">
          Business will{" "}
          <span className={statusColor}>
            {!data.is_successful && "not"} be successful
          </span>
        </p>
      </CardBody>
    </Card>
  );
};

export default SuccessChart;
