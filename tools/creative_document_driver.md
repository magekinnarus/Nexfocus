# Local creative document driver

The editor's **Local Agent driver** controls let a Director choose `inspect`,
`propose`, and/or `mutate`, enable a one hour local grant, and download its
capability JSON file. The CLI reads that file locally. It accepts only its
loopback URL, sends the bearer in the Authorization header, and never follows
redirects. Revoke the grant in the editor when finished.

```powershell
python tools/creative_document_driver.py --capability-file .\driver.json status
python tools/creative_document_driver.py --capability-file .\driver.json inspect --request .\inspect.json
python tools/creative_document_driver.py --capability-file .\driver.json propose --request .\proposal.json
python tools/creative_document_driver.py --capability-file .\driver.json mutate --request .\mutation.json
python tools/creative_document_driver.py --capability-file .\driver.json preview --expected-revision 4 --output .\preview.png
```

`inspect`, `propose`, and `mutate` read one strict JSON v1 envelope from a file
or stdin (`--request -`). All responses are deterministic compact JSON on
stdout; diagnostics go to stderr. Exit codes are `0` for success, `2` for a
service refusal, `3` for transport failure, and `4` for malformed local input
or response. Preview requires the revision returned by an inspect receipt and
checks the response's document and snapshot headers before writing the chosen
local output path.

Example read envelope:

```json
{
  "schemaVersion": 1,
  "commandId": "cmd-0123456789abcdef0123456789abcdef",
  "documentId": "doc-0123456789abcdef0123456789abcdef",
  "intent": "inspect",
  "expectedRevision": null,
  "commandType": "inspect_document",
  "targetIds": [],
  "coordinateSpace": "document",
  "payload": {},
  "transaction": null
}
```

Mutation and proposal envelopes pin an integer `expectedRevision` and include
`transaction` with opaque `transactionId`, `groupId`, and `phase: "commit"`.
Their `payload` must match the allowlisted W03 action shape. Agent import bytes,
arbitrary paths, network URLs, shell/code execution, and conversational commands
are not supported.

`targetIds` names the direct command subjects. Create commands use an empty
array; destination layers, parents, replacement IDs, and relational-context
references remain typed fields inside `payload`. A batch declares the ordered
union of its direct subcommand targets, and each transform subcommand names its
own target IDs. The service derives and checks this binding before preparation.
Receipts include the trusted `actorId` associated with the capability grant.
