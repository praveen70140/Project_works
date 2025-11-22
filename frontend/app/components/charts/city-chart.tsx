import { cityRecommendationApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { ResponsiveBar } from "@nivo/bar";
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
      Cuisine: payload.cuisine,
    }),
  }).then((r) => r.json());

const CityChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<cityRecommendationApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/rf_city_recommend`, payload],
    fetcher
  );

  if (isLoading)
    return (
      <div>
        <CircularProgress />
      </div>
    );
  if (!data) return <div>Not found</div>;

  const chartData = data.top_3_recommendations.map((e) => ({
    City: e.city,
    Probability: e.probability_percent,
  }));

  return (
    <Card className="col-span-3">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start absolute top-3">City Recommendations</h3>

        <div className="w-full h-96">
          <ResponsiveBar /* or Bar for fixed dimensions */
            data={chartData}
            keys={["Probability"]}
            indexBy="City"
            labelSkipWidth={12}
            colors={{ scheme: "accent" }}
            labelSkipHeight={12}
            axisBottom={{ legend: "City", legendOffset: 32 }}
            axisLeft={{ legend: "Probability (%age)", legendOffset: -40 }}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
          />{" "}
        </div>
      </CardBody>
    </Card>
  );
};

export default CityChart;
