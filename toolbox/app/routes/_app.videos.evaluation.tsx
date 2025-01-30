import React, { useState } from "react";
import fs from "fs";
import path from "path";
import { VideoGrid } from "~/components/video/video-grid";
import { Slider } from "~/components/ui/slider";
import { VideoCard } from "~/components/video/video-card";
import {
	type LoaderFunctionArgs,
	useLoaderData,
	useSearchParams,
} from "react-router";

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
			"evaluation",
		);
		const files = fs.existsSync(directoryPath)
			? fs.readdirSync(directoryPath, { recursive: true, encoding: "utf-8" })
			: [];

		return files.flatMap((file) => {
			const parts = file.match(/(\d+)_(\d+)(_?(\d+))?\.mp4/);
			if (parts) {
				const [, reward, episode, , step] = parts;
				if (!step) {
					return [];
				}

				return [
					{
						reward: +reward,
						episode: +episode,
						step: +step,
						videoSrc: `/public/vids/${run}/evaluation/${file}`,
						poster: "https://via.placeholder.com/480",
					},
				];
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

const EvaluationVideoCard = ({
	videoSrc,
	poster,
	reward,
	episode,
	step,
}: {
	videoSrc: string;
	poster: string;
	reward: number;
	episode: number;
	step: number;
}) => {
	return (
		<VideoCard videoSrc={videoSrc} poster={poster}>
			<div>
				<p>Reward: {reward}</p>
				<p>Episode: {episode}</p>
				<p>Step: {step}</p>
			</div>
		</VideoCard>
	);
};

export default function EvaluationComparisonPage() {
	const windowSize = 8;
	const [windowStart, setWindowStart] = useState(0);
	const [searchParams] = useSearchParams();
	let { run1, run2, run1Videos, run2Videos } = useLoaderData<typeof loader>();

	interface Run {
		step: number;
		episode: number;
	}
	const sortVideos = (a: Run, b: Run) => {
		if (a.step === b.step) {
			return a.episode - b.episode;
		}

		return a.step - b.step;
	};
	run1Videos.sort(sortVideos);
	run2Videos.sort(sortVideos);

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
			<h1 className="text-2xl font-bold mb-4">Evaluation Videos Comparison</h1>
			<Slider
				className="mb-8"
				min={0}
				max={Math.max(
					0,
					Math.max(run1Videos.length, run2Videos.length) - windowSize,
				)}
				defaultValue={[0]}
				onValueChange={handleSliderChange}
			/>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
				<div>
					<h2 className="text-xl font-bold mb-4">{run1}</h2>
					<VideoGrid>
						{visibleRun1Videos.map((video, index) => (
							<EvaluationVideoCard
								key={`eval1-${index}`}
								videoSrc={video.videoSrc}
								poster={video.poster}
								reward={video.reward}
								episode={video.episode}
								step={video.step}
							/>
						))}
					</VideoGrid>
				</div>
				<div>
					<h2 className="text-xl font-bold mb-4">{run2}</h2>
					<VideoGrid>
						{visibleRun2Videos.map((video, index) => (
							<EvaluationVideoCard
								key={`eval2-${index}`}
								videoSrc={video.videoSrc}
								poster={video.poster}
								reward={video.reward}
								episode={video.episode}
								step={video.step}
							/>
						))}
					</VideoGrid>
				</div>
			</div>
		</div>
	);
}
