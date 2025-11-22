import { monthRecommendationApi } from "@/app/types/api-types";
import { mainFormData } from "@/app/types/main-form-data";
import { Card, CardBody, CircularProgress } from "@heroui/react";
import { useFormContext } from "react-hook-form";
import useSWR from "swr";
import { ResponsiveLine } from "@nivo/line";
import { months } from "@/app/utils/time";

const fetcher = ([url, payload]: [url: string, payload: mainFormData]) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      year: payload.date.year,
      sales_qty: payload.salesQuantity,
      sales_amount: payload.salesAmount,
      Ratings: payload.rating,
      Cuisine: payload.cuisine,
      City: payload.city,
    }),
  }).then((r) => r.json());

const MonthChart = () => {
  const { getValues } = useFormContext<mainFormData>();
  const payload = getValues();
  const { data, error, isLoading } = useSWR<monthRecommendationApi>(
    [`${process.env.NEXT_PUBLIC_API_URL}/predict/rf_month_recommend`, payload],
    fetcher
  );

  if (isLoading)
    return (
      <div>
        <CircularProgress />
      </div>
    );
  if (!data) return <div>Not found</div>;

  const chartData = [
    {
      id: "Month",
      data: data.top_3_month_recommendations.map((e) => ({
        x: months[e.month - 1],
        y: e.probability_percent,
      })),
    },
  ];

  return (
    <Card className="col-span-3">
      <CardBody className="flex flex-col justify-center items-center">
        <h3 className="self-start absolute top-3">Month Recommendations</h3>

        <div className="w-full h-96">
          <ResponsiveLine /* or Line for fixed dimensions */
            data={chartData}
            margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
            yScale={{
              type: "linear",
              min: "auto",
              max: "auto",
              stacked: true,
              reverse: false,
            }}
            axisBottom={{ legend: "Months", legendOffset: 36 }}
            axisLeft={{ legend: "Probability (%age)", legendOffset: -40 }}
            pointSize={10}
            colors={{ scheme: "category10" }}
            pointColor={{ theme: "background" }}
            pointBorderWidth={2}
            pointBorderColor={{ from: "seriesColor" }}
            pointLabelYOffset={-12}
            enableTouchCrosshair={true}
            useMesh={true}
          />
        </div>
      </CardBody>
    </Card>
  );
};

export default MonthChart;
