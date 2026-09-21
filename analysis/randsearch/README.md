## Data used for selection vs test

| Dataset | Selection (pick top-1) | Test (reported) |
|---|---|---|
| countdown | 200 | 2000 |
| gsm8k | 200 | 1319 |
| math500 | 200 | 300 |

There is no overlap between the training and test data.



## Results
<table>
<thead><tr><th>Model</th><th>Dataset</th><th>Population</th><th>Decoding</th><th>Base</th><th>Members &gt; base</th><th>Top-1 gain</th><th>Best member gain</th></tr></thead>
<tbody>
<tr><td rowspan="4"><b>Qwen2.5-3B</b></td><td rowspan="2">countdown</td><td rowspan="2">412</td><td>greedy</td><td>0.148</td><td>25%</td><td>+1.5pp</td><td class="pos">+8.8pp</td></tr>
<tr><td>T=1.2</td><td>0.062</td><td>29%</td><td class="pos">+2.1pp</td><td class="pos">+2.4pp</td></tr>
<tr><td rowspan="2">gsm8k</td><td rowspan="2">412</td><td>greedy</td><td>0.812</td><td>5%</td><td>-0.7pp</td><td>+1.6pp</td></tr>
<tr><td>T=0.8</td><td>0.771</td><td>19%</td><td>-0.0pp</td><td class="pos">+2.1pp</td></tr>
<tr><td rowspan="2"><b>Qwen2.5-7B</b></td><td rowspan="2">countdown</td><td rowspan="2">216</td><td>greedy</td><td>0.322</td><td>38%</td><td class="pos">+9.8pp</td><td class="pos">+9.8pp</td></tr>
<tr><td>T=0.6</td><td>0.348</td><td>37%</td><td class="pos">+5.7pp</td><td class="pos">+5.7pp</td></tr>
<tr><td rowspan="6"><b>Qwen2.5-1.5B</b></td><td rowspan="2">countdown</td><td rowspan="2">412</td><td>greedy</td><td>0.122</td><td>24%</td><td>+1.5pp</td><td class="pos">+4.7pp</td></tr>
<tr><td>T=1.2</td><td>0.018</td><td>38%</td><td>+0.6pp</td><td>+0.9pp</td></tr>
<tr><td rowspan="2">gsm8k</td><td rowspan="2">412</td><td>greedy</td><td>0.607</td><td>33%</td><td class="pos">+4.5pp</td><td class="pos">+7.0pp</td></tr>
<tr><td>T=0.6</td><td>0.597</td><td>30%</td><td class="pos">+3.4pp</td><td class="pos">+3.9pp</td></tr>
<tr><td rowspan="2">math500</td><td rowspan="2">216</td><td>greedy</td><td>0.463</td><td>26%</td><td>+1.3pp</td><td class="pos">+3.3pp</td></tr>
<tr><td>T=1.2</td><td>0.129</td><td>25%</td><td class="pos">+3.6pp</td><td class="pos">+3.6pp</td></tr>
<tr><td rowspan="6"><b>Olmo-3-7B</b></td><td rowspan="2">countdown</td><td rowspan="2">216</td><td>greedy</td><td>0.687</td><td>2%</td><td class="neg">-2.7pp</td><td>+0.3pp</td></tr>
<tr><td>T=1.0</td><td>0.621</td><td>50%</td><td>+1.6pp</td><td class="pos">+3.7pp</td></tr>
<tr><td rowspan="2">gsm8k</td><td rowspan="2">216</td><td>greedy</td><td>0.871</td><td>50%</td><td>-0.7pp</td><td>+1.1pp</td></tr>
<tr><td>T=1.0</td><td>0.860</td><td>59%</td><td>+0.0pp</td><td>+1.7pp</td></tr>
<tr><td rowspan="2">math500</td><td rowspan="2">216</td><td>greedy</td><td>0.617</td><td>34%</td><td class="neg">-2.0pp</td><td class="pos">+4.3pp</td></tr>
<tr><td>T=0.6</td><td>0.581</td><td>73%</td><td class="pos">+2.0pp</td><td class="pos">+6.4pp</td></tr>
</tbody>
</table>