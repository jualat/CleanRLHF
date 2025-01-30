import React, { useState } from "react";
import fs from "fs";
import { VideoGrid } from "~/components/video/video-grid";
import path from "path";
import {
	type LoaderFunctionArgs,
	useLoaderData,
	useSearchParams,
} from "react-router";
import { Slider } from "~/components/ui/slider";
import { VideoCard } from "~/components/video/video-card";

export const loader = async ({ request }: LoaderFunctionArgs) => {
	const url = new URL(request.url);
	const run1 = url.searchParams.get("run1");
	const run2 = url.searchParams.get("run2");

	const loadVideos = (run: string | null) => {
		if (!run) return [];

		const directoryPath = path.join(
			process.cwd(),
			"./public/vids",
			run,
			"trajectories",
		);
		const files = fs.existsSync(directoryPath)
			? fs.readdirSync(directoryPath, { recursive: true, encoding: "utf-8" })
			: [];

		return files
			.filter((file) => file.endsWith(".mp4"))
			.flatMap((file) => {
				const parts = file.match(/trajectory_(\d+)_(\d+)_(\d+)\.mp4/);
				if (parts) {
					const [, startIndex, endIndex, step] = parts;
					return {
						startIndex: +startIndex,
						endIndex: +endIndex,
						step: +step,
						videoSrc: `/vids/${run}/trajectories/${file}`,
						poster: "https://via.placeholder.com/480",
					};
				}
				return [];
			});
	};

	const run1Videos = loadVideos(run1);
	const run2Videos = loadVideos(run2);

	return {
		run1,
		run2,
		run1Videos,
		run2Videos,
	};
};

const ReplayBufferVideoCard = ({
	videoSrc,
	poster,
	startIndex,
	endIndex,
	step,
}: {
	videoSrc: string;
	poster: string;
	startIndex: number;
	endIndex: number;
	step: number;
}) => {
	return (
		<VideoCard videoSrc={videoSrc} poster={poster}>
			{startIndex} - {endIndex}
			<br />
		</VideoCard>
	);
};

export default function Page() {
	const windowSize = 8;
	const [windowStart, setWindowStart] = useState(0);
	const [searchParams] = useSearchParams();
	let { run1, run2, run1Videos, run2Videos } = useLoaderData<typeof loader>();
	run1Videos = run1Videos.sort((a, b) => a.startIndex - b.startIndex);
	run2Videos = run2Videos.sort((a, b) => a.startIndex - b.startIndex);

	const [visibleRun1Videos, setVisibleRun1Videos] = useState(
		run1Videos.slice(0, windowSize),
	);
	const [visibleRun2Videos, setVisibleRun2Videos] = useState(
		run2Videos.slice(0, windowSize),
	);

	const handleSliderChange = (value: number[]) => {
		const newStart = value[0];
		setWindowStart(newStart);
		setVisibleRun1Videos(run1Videos.slice(newStart, newStart + windowSize));
		setVisibleRun2Videos(run2Videos.slice(newStart, newStart + windowSize));
	};

	return (
		<div className="container mx-auto py-8 px-8">
			<h1 className="text-2xl font-bold mb-4">
				Replay Buffer Videos Comparison
			</h1>
			<Slider
				className="mb-8"
				min={0}
				max={Math.max(
					0,
					Math.max(run1Videos.length, run2Videos.length) - windowSize,
				)}
				defaultValue={[windowStart]}
				onValueChange={handleSliderChange}
			/>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
				<div>
					<h2 className="text-xl font-bold mb-4">{run1}</h2>
					<VideoGrid>
						{visibleRun1Videos.map((video, index) => (
							<ReplayBufferVideoCard
								key={`t1-${index}`}
								videoSrc={video.videoSrc}
								poster={video.poster}
								startIndex={video.startIndex}
								endIndex={video.endIndex}
								step={video.step}
							/>
						))}
					</VideoGrid>
				</div>
				<div>
					<h2 className="text-xl font-bold mb-4">{run2}</h2>
					<VideoGrid>
						{visibleRun2Videos.map((video, index) => (
							<ReplayBufferVideoCard
								key={`t2-${index}`}
								videoSrc={video.videoSrc}
								poster={video.poster}
								startIndex={video.startIndex}
								endIndex={video.endIndex}
								step={video.step}
							/>
						))}
					</VideoGrid>
				</div>
			</div>
		</div>
	);
}
